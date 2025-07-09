import argparse
import wandb
import pandas as pd
import numpy as np
import logging
import os
import sys
import subprocess
import re
import shutil
import yaml  # For parsing config.yaml

# Use non-interactive backend to avoid hanging on headless servers
import matplotlib
matplotlib.use("Agg", force=True)  # must be set before importing pyplot
import matplotlib.pyplot as plt
# wandb_mpl.py
from contextlib import contextmanager
from cycler import cycler

# --- default W&B colour cycle -------------------------------------------------
_WANDB_PALETTE = [
    "#F15854", "#5DA5DA", "#60BD68", "#F17CB0",
    "#B2912F", "#B276B2", "#DECF3F", "#4D4D4D",
]

def _build_rc(font_scale=1.0, linewidth=2.0, figsize=(7, 2.5), dpi=160,
              palette=None, grid=True):
    """Return an rcParams-dict that mimics W&B's dashboard style."""
    palette = palette or _WANDB_PALETTE
    base_fs = 10 * font_scale          # 10 pt is W&B's default body size
    return {
        # colours & lines
        "axes.prop_cycle"  : cycler("color", palette),
        "lines.linewidth"  : linewidth,
        "lines.markersize" : 6,

        # grid & spines
        "axes.grid"        : grid,
        "grid.color"       : "#E5E5E5",
        "grid.linewidth"   : 1.0,
        "grid.alpha"       : 1.0,
        "grid.linestyle"   : "-",
        "axes.spines.top"  : False,
        "axes.spines.right": False,

        # fonts
        "font.family"      : "sans-serif",
        "font.size"        : base_fs,
        "axes.titlesize"   : base_fs * 1.2,
        "axes.labelsize"   : base_fs,
        "xtick.labelsize"  : base_fs * 0.9,
        "ytick.labelsize"  : base_fs * 0.9,
        "legend.fontsize"  : base_fs * 0.9,

        # figure geometry
        "figure.figsize"   : figsize,
        "figure.dpi"       : dpi,
    }

@contextmanager
def wandb_style(**kwargs):
    """
    Context manager that applies a W&B-ish style.
    
    Parameters (all optional)
    -------------------------
    font_scale : float  (default 1.0)  – multiplies every font size
    linewidth  : float  (default 2.0)  – default line thickness
    figsize    : (w,h)  (default (7,2.5) inches)
    dpi        : int    (default 160)
    palette    : list[str] – custom colour cycle
    grid       : bool   – turn grid on/off (default True)
    """
    rc = _build_rc(**kwargs)
    with plt.rc_context(rc):
        yield



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sync_and_get_run_path(run_dir):
    """
    Syncs a local wandb run directory to the cloud and extracts the run path.
    Crashes if syncing fails or path isn't found.
    """
    assert os.path.isdir(run_dir), f"Run directory not found: {run_dir}"
    assert shutil.which('wandb'), "'wandb' command not found. Is wandb installed and in your PATH?"

    logging.info(f"Syncing run directory: {run_dir}")
    result = subprocess.run(
        ['wandb', 'sync', run_dir],
        capture_output=True,
        text=True,
        check=True  # Will raise CalledProcessError on non-zero exit
    )
    
    output = result.stdout + result.stderr
    logging.info("Sync command output:\n" + output)

    match = re.search(r"https://wandb.ai/([^/]+)/([^/]+)/runs/([^/ ]+)", output)
    assert match, "Could not find run path in wandb sync output."
    
    entity, project, run_id = match.groups()
    run_path = f"{entity}/{project}/{run_id}"
    logging.info(f"Successfully synced. Found run path: {run_path}")
    return run_path


def fetch_run_data(run_path):
    """
    Fetches run data using the wandb API.
    Crashes if the run is not found.
    """
    logging.info(f"Attempting to fetch run data for: {run_path}")
    api = wandb.Api()
    run = api.run(run_path)
    assert run, f"Run '{run_path}' not found. Are you logged in and is the path correct?"
    logging.info("Run data fetched successfully.")
    return run


# -----------------------------------------------------------------------------
# Modular helpers
# -----------------------------------------------------------------------------

def get_step_column(df):
    """Determine which step column exists in the DataFrame."""
    if '_step' in df.columns:
        return '_step'
    elif 'step' in df.columns:
        return 'step'
    else:
        raise ValueError(f"Neither '_step' nor 'step' column found in the DataFrame: {df.columns}")


def collect_run_histories(run_dirs, metric: str, *, entity: str, project: str):
    """Return list of DataFrames containing [step_col, metric] for each run directory."""
    histories = []
    api = wandb.Api()

    for rd in run_dirs:
        run_id = os.path.basename(rd).split('-')[-1]

        cfg_entity, cfg_project = extract_wandb_info(rd)
        if cfg_project is not None and cfg_project != project:
            logging.info(
                f"Skipping run {run_id}: config project '{cfg_project}' does not match requested '{project}'"
            )
            continue
        logging.info(f"processing run {run_id} cfg_entity {cfg_entity} cfg_project {cfg_project}")

        effective_entity = cfg_entity or entity
        effective_project = cfg_project or project

        run_path_guess = f"{effective_entity}/{effective_project}/{run_id}"

        try:
            run = fetch_run_data(run_path_guess)
            logging.info(f"Loaded run from cloud: {run_path_guess}")
        except (wandb.errors.CommError, wandb.errors.Error):
            # logging.warning(f"Run {run_id} not found in cloud – syncing from {rd}")
            # actual_run_path = sync_and_get_run_path(rd)
            # run = fetch_run_data(actual_run_path)
            continue

        try:
            full_hist_df = run.history()
            
            # Check if DataFrame is empty or corrupted
            if full_hist_df.empty or len(full_hist_df.columns) == 0:
                logging.warning(f"Skipping run {run_id}: empty or corrupted data")
                continue
            
            # Check if the metric exists
            if metric not in full_hist_df.columns:
                logging.warning(f"Skipping run {run_id}: metric '{metric}' not found in columns: {list(full_hist_df.columns)}")
                continue
            
            step_col = get_step_column(full_hist_df)
            logging.info(f"Using step column '{step_col}' for run {run_id}")
            hist_df = full_hist_df[[step_col, metric]]
            hist_df.dropna(subset=[metric], inplace=True)
            
            # Check if we have any data left after dropping NaNs
            if hist_df.empty:
                logging.warning(f"Skipping run {run_id}: no valid data points after removing NaN values")
                continue
                
            # Standardize column name to '_step' for consistency
            hist_df = hist_df.rename(columns={step_col: '_step'})
            histories.append(hist_df)
            
        except Exception as e:
            logging.warning(f"Skipping run {run_id}: error processing data - {str(e)}")
            continue

    if not histories:
        logging.error("No valid run histories found. All runs were skipped due to corruption or missing data.")
        
    return histories


def aggregate_histories(histories: list[pd.DataFrame], metric: str):
    """Aggregate multiple run histories; return (steps, mean, lower, upper, label)."""
    if len(histories) == 1:
        df = histories[0].sort_values('_step')
        steps = df['_step'].to_numpy()
        mean = df[metric].to_numpy()
        return steps, mean, None, None, f'{metric} (n=1)'

    merged = None
    for idx, df in enumerate(histories):
        df = df.set_index('_step').rename(columns={metric: f'run{idx}'})
        merged = df if merged is None else merged.join(df, how='outer')

    steps = merged.index.to_numpy()
    values = merged.to_numpy(dtype=float)

    mean = np.nanmean(values, axis=1)
    sem = np.nanstd(values, axis=1, ddof=1) / np.sqrt(np.sum(~np.isnan(values), axis=1))

    return steps, mean, mean - sem, mean + sem, f"{metric} (n={len(histories)})"


def plot_history(steps, mean, lower, upper, label, *, project: str, metric: str, n_runs: int):
    with wandb_style():
        plt.figure()
        plt.plot(steps, mean, '-', label=label)
        if lower is not None and upper is not None:
            plt.fill_between(steps, lower, upper, alpha=0.3)

        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        title_suffix = f" ({n_runs} seeds)" if n_runs > 1 else ""
        plt.title(f"{metric} – project '{project}'{title_suffix}")
        plt.xlabel("Training Step")
        plt.ylabel("Mean Return")
        plt.legend()
        os.makedirs("figures", exist_ok=True)
        metric_safe = metric.replace("/", "_")
        plt.savefig(f"figures/plot_{project}_{metric_safe}.png",
                    bbox_inches='tight',
                    dpi=160)
        plt.close()


def plot_multiple_metrics(results: list[dict], *, project: str):
    """Plot multiple metrics on one figure."""
    with wandb_style():
        plt.figure()
        for res in results:
            steps = res['steps']
            mean = res['mean']
            lower = res['lower']
            upper = res['upper']
            label = res['label']
            plt.plot(steps, mean, '-', label=label)
            if lower is not None and upper is not None:
                plt.fill_between(steps, lower, upper, alpha=0.2)

        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        plt.title(f"'{project}'")
        plt.xlabel("Training Step")
        plt.ylabel("Value")
        plt.legend()
        os.makedirs("figures", exist_ok=True)
        metrics_safe = "__".join([m['metric'].replace('/', '_') for m in results])[:100]
        plt.savefig(f"figures/plot_{project}_combined_{metrics_safe}.png", bbox_inches='tight', dpi=160)
        plt.close()


def extract_wandb_info(run_dir):
    """Extract entity and project names from the run's config.yaml (if present).

    Returns (entity, project) – either may be None if not found.
    """
    cfg_path = os.path.join(run_dir, 'files', 'config.yaml')
    if not os.path.isfile(cfg_path):
        return None, None

    try:
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        logging.debug(f"Failed to parse {cfg_path}: {e}")
        return None, None

    # config.yaml can store primitives or dicts with a 'value' field
    def _resolve(val):
        if isinstance(val, dict):
            return val.get('value')
        return val

    entity = _resolve(cfg.get('wandb_entity') or cfg.get('entity'))
    project = _resolve(cfg.get('wandb_project') or cfg.get('project'))
    return entity, project


def main():
    """
    Main function to parse arguments, sync run, and fetch data.
    """
    parser = argparse.ArgumentParser(
        description="Fetch and plot data from a wandb run. Tries to fetch directly first, falls back to syncing if needed."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="wandb",
        help="Either (i) path to one local wandb run directory (run-XXXX) or (ii) a directory that contains several such run-* sub-folders produced by multiple seeds."
    )
    parser.add_argument(
        "--entity", 
        type=str, 
        default=None, 
        help="Wandb entity. Defaults to the logged-in user."
    )
    parser.add_argument(
        "--project", 
        type=str, 
        default="sensor-dropout-2", 
        help="Wandb project name. Important for fetching without syncing."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["full_eval_return/mean", "eval/mean_return"],
        help="One or more metric keys to fetch (e.g. full_eval_return/mean train/ep_return). A separate PNG will be produced per metric.",
    )

    args = parser.parse_args()

    # Change to the script's directory to ensure relative paths work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # ---------------------------------------------------------------------
    # Decide whether we are dealing with one run (single seed) or many runs.
    # ---------------------------------------------------------------------
    run_dirs: list[str] = []

    if os.path.basename(args.run_dir).startswith("run-"):
        # Specific run directory – treat as single
        run_dirs = [args.run_dir]
    else:
        # Aggregate: look for sub-folders run-* inside provided directory
        run_dirs = [
            os.path.join(args.run_dir, d)
            for d in os.listdir(args.run_dir)
            if d.startswith("run-") and os.path.isdir(os.path.join(args.run_dir, d))
        ]
        assert run_dirs, f"No run-* subdirectories found inside {args.run_dir}"

    logging.info(f"Found {len(run_dirs)} run directories to process.")

    # ---------------------------------------------------------------------
    # Collect → aggregate → plot   for each requested metric
    # ---------------------------------------------------------------------
    entity = args.entity or wandb.Api().default_entity

    aggregated = []  # store results for each metric to combine later

    for metric in args.metrics:
        logging.info(f"Processing metric '{metric}' ...")
        histories = collect_run_histories(run_dirs, metric, entity=entity, project=args.project)
        steps, mean, lower, upper, label = aggregate_histories(histories, metric)
        aggregated.append({
            'metric': metric,
            'steps': steps,
            'mean': mean,
            'lower': lower,
            'upper': upper,
            'label': label,
            'n_runs': len(histories),
        })

    # Plot – single combined figure if >1 metric, else single plot
    if not aggregated:
        logging.error("No data collected – exiting.")
        return

    if len(aggregated) == 1:
        res = aggregated[0]
        plot_history(
            res['steps'], res['mean'], res['lower'], res['upper'], res['label'],
            project=args.project, metric=res['metric'], n_runs=res['n_runs'],
        )
    else:
        plot_multiple_metrics(aggregated, project=args.project)


if __name__ == "__main__":
    main() 