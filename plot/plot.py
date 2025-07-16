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

def get_plot_filename(mode, projects, metrics):
    """
    Generates a standardized plot filename.

    Args:
        mode (str): 'comparison', 'multiple_metrics', or 'single'.
        projects (list[str]): List of project names.
        metrics (list[str]): List of metric names.

    Returns:
        str: The generated filename for the plot.
    """
    def sanitize(s):
        return s.replace('/', '_')

    if mode == 'comparison':
        # Comparing multiple projects for a single metric
        projects_safe = "__".join(sorted([sanitize(p) for p in projects]))[:100]
        metric_safe = sanitize(metrics[0])
        return f"figures/plot_compare_projects_{projects_safe}_{metric_safe}.png"
    elif mode == 'multiple_metrics':
        # Single project with multiple metrics
        project_safe = sanitize(projects[0])
        metrics_safe = "__".join([sanitize(m) for m in metrics])[:100]
        return f"figures/plot_{project_safe}_combined_{metrics_safe}.png"
    elif mode == 'single':
        # Single project, single metric
        project_safe = sanitize(projects[0])
        metric_safe = sanitize(metrics[0])
        return f"figures/plot_{project_safe}_{metric_safe}.png"
    else:
        raise ValueError(f"Unknown plot mode: {mode}")

# --- default W&B colour cycle -------------------------------------------------
_WANDB_PALETTE = [
    "#FF7F00", "#FF1493", "#87CEEB",  
    "#F15854", "#5DA5DA", "#60BD68", "#F17CB0",
    "#B2912F", "#B276B2", "#DECF3F", "#4D4D4D",
]

def _build_rc(font_scale=1.0, linewidth=2.0, figsize=(7, 3.5), dpi=160,
              palette=None, grid=True):
    """Return an rcParams-dict that mimics W&B's dashboard style."""
    palette = palette or _WANDB_PALETTE
    base_fs = 10 * font_scale          # 10 pt is W&B's default body size
    gray = "#777777"
    return {
        # colours & lines
        "axes.prop_cycle"  : cycler("color", palette),
        "lines.linewidth"  : linewidth,
        "lines.markersize" : 6,

        # grid & spines
        "axes.grid"        : False, # We'll add horizontal grid manually
        "grid.color"       : "#E5E5E5",
        "grid.linewidth"   : 1.0,
        "grid.alpha"       : 1.0,
        "grid.linestyle"   : "-",
        "axes.spines.top"  : False,
        "axes.spines.right": False,
        "axes.edgecolor"   : gray,

        # fonts
        "text.color"       : gray,
        "font.family"      : "Inter",
        "font.size"        : base_fs,
        "axes.titlesize"   : base_fs * 1.2,
        "axes.titleweight" : "bold",
        "axes.titlecolor"  : "black",
        "axes.labelsize"   : base_fs,
        "axes.labelcolor"  : gray,
        "xtick.labelsize"  : base_fs * 0.9,
        "ytick.labelsize"  : base_fs * 0.9,
        "xtick.color"      : gray,
        "ytick.color"      : gray,
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
    # logging.info(f"Attempting to fetch run data for: {run_path}")
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
    return "global_step"
    # if '_step' in df.columns:
    #     return '_step'
    # elif 'step' in df.columns:
    #     return 'step'
    # else:
    #     raise ValueError(f"Neither '_step' nor 'step' column found in the DataFrame: {df.columns}")


def collect_run_histories(run_dirs, metric: str, *, entity: str, project: str):
    """Return list of DataFrames containing [step_col, metric] for each run directory."""
    histories = []
    api = wandb.Api()

    for rd in run_dirs:
        run_id = os.path.basename(rd).split('-')[-1]

        cfg_entity, cfg_project = extract_wandb_info(rd)
        # if cfg_project is not None and cfg_project != project:
        if cfg_project != project:
            # logging.debug(
            #     f"Skipping run {run_id}: config project '{cfg_project}' does not match requested '{project}'"
            # )
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
            # Get ALL data points by setting samples to a high number
            full_hist_df = run.history(samples=100000000) ## otherwise, the default is like 500 to save memory
            
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
            # Create a copy to avoid SettingWithCopyWarning
            hist_df = full_hist_df[[step_col, metric]].copy()
            hist_df.dropna(subset=[metric], inplace=True)
            
            # Log data statistics for debugging
            logging.info(f"Run {run_id}: fetched {len(full_hist_df)} total rows, {len(hist_df)} rows with valid {metric} data")
            
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
    else:
        logging.info(f"Successfully collected {len(histories)} run histories for metric '{metric}'")
        
    return histories


def aggregate_histories(histories: list[pd.DataFrame], metric: str):
    """Aggregate multiple run histories; return (steps, mean, lower, upper, label)."""
    logging.info(f"Aggregating {len(histories)} histories for metric '{metric}'")
    
    if len(histories) == 1:
        df = histories[0].sort_values('_step')
        steps = df['_step'].to_numpy()
        mean = df[metric].to_numpy()
        logging.info(f"Single run: {len(steps)} data points")
        return steps, mean, None, None, f'{metric} (N=1)'

    merged = None
    for idx, df in enumerate(histories):
        df = df.set_index('_step').rename(columns={metric: f'run{idx}'})
        merged = df if merged is None else merged.join(df, how='outer')

    steps = merged.index.to_numpy()
    values = merged.to_numpy(dtype=float)

    mean = np.nanmean(values, axis=1)
    sem = np.nanstd(values, axis=1, ddof=1) / np.sqrt(np.sum(~np.isnan(values), axis=1))

    return steps, mean, mean - sem, mean + sem, f"{metric} (N={len(histories)})"


def plot_history(steps, mean, lower, upper, label, *, filename: str, project: str, metric: str, n_runs: int, ymin=None, ymax=None):
    with wandb_style():
        plt.figure()
        plt.grid(axis='y', color="#E5E5E5", linestyle='-', linewidth=1.0, alpha=1.0)
        line, = plt.plot(steps, mean, '-', label=label)
        if lower is not None and upper is not None:
            plt.fill_between(steps, lower, upper, alpha=0.3, color=line.get_color())

        plt.ylim(bottom=ymin, top=ymax)

        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        plt.xlim(left=steps.min(), right=steps.max())
        plt.locator_params(axis='x', nbins=6)
        title_suffix = f" ({n_runs} seeds)" if n_runs > 1 else ""
        plt.title(f"{metric} – project '{project}'{title_suffix}")
        plt.xlabel("global_step", loc='right')
        # plt.ylabel("Mean Return") # Removed to match wandb UI
        legend = plt.legend(loc="lower right", frameon=False)
        # Match legend text color to line color
        for text in legend.get_texts():
            text.set_color(line.get_color())

        os.makedirs("figures", exist_ok=True)
        print(f"Saving figure to {filename}")
        plt.savefig(filename,
                    bbox_inches='tight',
                    dpi=160)
        plt.close()


def plot_comparison(results: list[dict], *, title: str, ylabel: str, filename: str, ymin=None, ymax=None):
    """Plot multiple curves on one figure."""
    with wandb_style():
        plt.figure()
        plt.grid(axis='y', color="#E5E5E5", linestyle='-', linewidth=1.0, alpha=1.0)
        for res in results:
            steps = res['steps']
            mean = res['mean']
            lower = res['lower']
            upper = res['upper']
            label = res['label']
            line, = plt.plot(steps, mean, '-', label=label)
            if lower is not None and upper is not None:
                plt.fill_between(steps, lower, upper, alpha=0.2, color=line.get_color())

        plt.ylim(bottom=ymin, top=ymax)

        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        plt.xlim(left=min([r['steps'].min() for r in results if len(r['steps']) > 0]), right=max([r['steps'].max() for r in results if len(r['steps']) > 0]))
        plt.locator_params(axis='x', nbins=6)
        plt.title(title, fontweight='bold')
        plt.xlabel("global_step", loc='right')
        # plt.ylabel(ylabel) # Removed to match wandb UI
        legend = plt.legend(loc="lower right", frameon=False)
        # Match legend text color to line color
        for line, text in zip(legend.get_lines(), legend.get_texts()):
            text.set_color(line.get_color())

        os.makedirs("figures", exist_ok=True)
        plt.savefig(filename, bbox_inches='tight', dpi=160)
        plt.close()


def plot_multiple_metrics(results: list[dict], *, project: str, filename: str, ymin=None, ymax=None):
    """Plot multiple metrics on one figure."""
    with wandb_style():
        plt.figure()
        plt.grid(axis='y', color="#E5E5E5", linestyle='-', linewidth=1.0, alpha=1.0)
        for res in results:
            steps = res['steps']
            mean = res['mean']
            lower = res['lower']
            upper = res['upper']
            label = res['label']
            line, = plt.plot(steps, mean, '-', label=label)
            if lower is not None and upper is not None:
                plt.fill_between(steps, lower, upper, alpha=0.2, color=line.get_color())

        plt.ylim(bottom=ymin, top=ymax)

        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
        plt.xlim(left=min([r['steps'].min() for r in results if len(r['steps']) > 0]), right=max([r['steps'].max() for r in results if len(r['steps']) > 0]))
        plt.locator_params(axis='x', nbins=6)
        plt.title(f"{project}")
        plt.xlabel("global_step", loc='right')
        # plt.ylabel("Value") # Removed to match wandb UI
        legend = plt.legend(loc="lower right", frameon=False)
        # Match legend text color to line color
        for line, text in zip(legend.get_lines(), legend.get_texts()):
            text.set_color(line.get_color())

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
        "--projects", 
        nargs="+",
        required=True, 
        help="One or more Wandb project names to plot."
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["full_eval_return/env_mean/mean_return"],
        help="One or more metric keys to fetch (e.g. full_eval_return/mean). A comparison plot is made per metric.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Set y-axis lower limit.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Set y-axis upper limit.",
    )

    args = parser.parse_args()

    # Resolve run_dir to an absolute path *before* changing directory
    args.run_dir = os.path.abspath(args.run_dir)

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
        if not run_dirs:
            logging.warning(f"No run-* subdirectories found inside {args.run_dir}")

    logging.info(f"Found {len(run_dirs)} run directories to process.")

    # ---------------------------------------------------------------------
    # Collect → aggregate → plot   for each requested metric
    # ---------------------------------------------------------------------
    entity = args.entity or wandb.Api().default_entity

    # New logic: handle multiple projects and/or multiple metrics
    if len(args.projects) > 1 and len(args.metrics) > 1:
        # Case: Multiple projects AND multiple metrics - create one combined plot
        logging.info("--- Processing multiple projects and multiple metrics ---")
        all_results = []
        
        for project in args.projects:
            for metric in args.metrics:
                logging.info(f"  Fetching data for project: '{project}' with metric: '{metric}'")
                histories = collect_run_histories(run_dirs, metric, entity=entity, project=project)
                
                if not histories:
                    logging.warning(f"  No valid histories found for project '{project}' with metric '{metric}'")
                    continue

                steps, mean, lower, upper, _ = aggregate_histories(histories, metric)
                
                # Create a label that includes both project and metric
                label = f"{project}_{metric} (N={len(histories)})"

                all_results.append({
                    'metric': metric, 'steps': steps, 'mean': mean, 'lower': lower, 'upper': upper,
                    'label': label, 'n_runs': len(histories),
                })
        
        if not all_results:
            logging.error("No data collected for any project-metric combination. Skipping plot.")
            return
        
        filename = get_plot_filename('comparison', args.projects, args.metrics)
        plot_comparison(all_results, title="Multiple Projects and Metrics", ylabel="Value", filename=filename, ymin=args.ymin, ymax=args.ymax)
        print(f"Generated plot: {filename}")
        return
        
    elif len(args.projects) > 1:
        # Case: Multiple projects, single metric (or one metric per plot)
        for metric in args.metrics:
            logging.info(f"--- Processing metric: '{metric}' ---")
            results = []
            for project in args.projects:
                logging.info(f"  Fetching data for project: '{project}'")
                histories = collect_run_histories(run_dirs, metric, entity=entity, project=project)
                
                if not histories:
                    logging.warning(f"  No valid histories found for project '{project}' with metric '{metric}'")
                    continue

                steps, mean, lower, upper, _ = aggregate_histories(histories, metric)
                
                # Create a concise label from the project name with number of runs
                label = f"{project} (N={len(histories)})"

                results.append({
                    'metric': metric, 'steps': steps, 'mean': mean, 'lower': lower, 'upper': upper,
                    'label': label, 'n_runs': len(histories),
                })
            
            if not results:
                logging.error(f"No data collected for metric '{metric}' across any project. Skipping plot.")
                continue
            
            filename = get_plot_filename('comparison', args.projects, [metric])
            plot_comparison(results, title=metric, ylabel=metric, filename=filename, ymin=args.ymin, ymax=args.ymax)
            print(f"Generated plot: {filename}")
        
        return # Exit after handling multi-project comparison

    # --- Original logic for single project ---
    project = args.projects[0]
    aggregated = []  # store results for each metric to combine later

    for metric in args.metrics:
        logging.info(f"Processing metric '{metric}' ...")
        histories = collect_run_histories(run_dirs, metric, entity=entity, project=project)
        if not histories:
            logging.warning(f"No histories found for '{project}' with metric '{metric}'")
            continue
        steps, mean, lower, upper, _ = aggregate_histories(histories, metric)
        # When plotting multiple metrics, include metric name in label for clarity
        if len(args.metrics) > 1:
            label = f"{metric} (N={len(histories)})"
        else:
            label = f"{project} (N={len(histories)})" # Match wandb UI with run count
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
        filename = get_plot_filename('single', [project], [res['metric']])
        plot_history(
            res['steps'], res['mean'], res['lower'], res['upper'], res['label'],
            filename=filename, project=project, metric=res['metric'], n_runs=res['n_runs'], ymin=args.ymin, ymax=args.ymax,
        )
        print(f"Generated plot: {filename}")
    else:
        metrics = [res['metric'] for res in aggregated]
        filename = get_plot_filename('multiple_metrics', [project], metrics)
        plot_multiple_metrics(aggregated, project=project, filename=filename, ymin=args.ymin, ymax=args.ymax)
        print(f"Generated plot: {filename}")


if __name__ == "__main__":
    main() 