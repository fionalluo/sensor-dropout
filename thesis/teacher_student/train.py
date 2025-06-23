import os
import sys
import time
import random
import argparse
import warnings
import pathlib
import importlib
import re
from functools import partial as bind
from types import SimpleNamespace

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import ruamel.yaml

# Add thesis directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import embodied package and its components
import embodied
from embodied import wrappers
from embodied.core import Path, Flags, Config

# Import teacher-student components
from thesis.teacher_student.teacher import TeacherPolicy
from thesis.teacher_student.student import StudentPolicy
from thesis.teacher_student.bc import BehavioralCloning
from thesis.teacher_student.encoder import DualEncoder

def dict_to_namespace(d):
    """Convert a dictionary to a SimpleNamespace recursively."""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument('--configs', type=str, nargs='+', default=[], help="Which named configs to apply.")
    parser.add_argument('--seed', type=int, default=None, help="Override seed manually.")
    return parser.parse_args()

def load_config(argv=None):
    configs = ruamel.yaml.YAML(typ='safe').load(
        (embodied.Path(__file__).parent / 'config.yaml').read())
    
    parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
    config_dict = embodied.Config(configs['defaults'])

    for name in parsed.configs:
        config_dict = config_dict.update(configs[name])
    config_dict = embodied.Flags(config_dict).parse(other)
    
    # Convert to SimpleNamespace
    config = dict_to_namespace(config_dict)
    print(config)

    return config

def make_envs(config, num_envs):
    suite, task = config.task.split('_', 1)
    ctors = []
    for index in range(num_envs):
        ctor = lambda: make_env(config)
        if hasattr(config, 'envs') and hasattr(config.envs, 'parallel') and config.envs.parallel != 'none':
            ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
        if hasattr(config, 'envs') and hasattr(config.envs, 'restart') and config.envs.restart:
            ctor = bind(wrappers.RestartOnException, ctor)
        ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(hasattr(config, 'envs') and hasattr(config.envs, 'parallel') and config.envs.parallel != 'none'))

def make_env(config, **overrides):
    suite, task = config.task.split('_', 1)
    if "TrailEnv" in task or "GridBlindPick" or "LavaTrail" in task:
        import trailenv

    ctor = {
        'dummy': 'embodied.envs.dummy:Dummy',
        'gym': 'embodied.envs.from_gym:FromGym',
        'gymnasium': 'embodied.envs.from_gymnasium:FromGymnasium',
        'dm': 'embodied.envs.from_dmenv:FromDM',
        'crafter': 'embodied.envs.crafter:Crafter',
        'dmc': 'embodied.envs.dmc:DMC',
        'atari': 'embodied.envs.atari:Atari',
        'dmlab': 'embodied.envs.dmlab:DMLab',
        'minecraft': 'embodied.envs.minecraft:Minecraft',
        'loconav': 'embodied.envs.loconav:LocoNav',
        'pinpad': 'embodied.envs.pinpad:PinPad',
        'robopianist': 'embodied.envs.robopianist:RoboPianist'
    }[suite]
    if isinstance(ctor, str):
        module, cls = ctor.split(':')
        module = importlib.import_module(module)
        ctor = getattr(module, cls)
    kwargs = getattr(config.env, suite, {})
    kwargs.update(overrides)
    if suite == 'robopianist':
        # kwargs.update({
        # 'record': config.run.script == 'eval_only'  # record in eval only for now (single environment)
        # })
        render_image = False
        if 'Pixel' in task:
            task = task.replace('Pixel', '')
        render_image = True
        kwargs.update({'render_image': render_image})

    env = ctor(task, **kwargs)
    return wrap_env(env, config)

def wrap_env(env, config):
    args = getattr(config, 'wrapper', {})
    for name, space in env.act_space.items():
        if name == 'reset':
            continue
        elif space.discrete:
            env = wrappers.OneHotAction(env, name)
        elif hasattr(args, 'discretize') and args.discretize:
            env = wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = wrappers.NormalizeAction(env, name)

    env = wrappers.ExpandScalars(env)

    if hasattr(args, 'length') and args.length:
        env = wrappers.TimeLimit(env, args.length, getattr(args, 'reset', True))
    if hasattr(args, 'checks') and args.checks:
        env = wrappers.CheckSpaces(env)

    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)

    return env

def process_video_frames(frames, key):
    """Process frames for video logging following exact format requirements."""
    if len(frames.shape) == 3:  # Single image [H, W, C]
        # Check if the last dimension is 3 (RGB image) and the maximum value is greater than 1
        if frames.shape[-1] == 3 and np.max(frames) > 1:
            return frames  # Directly pass the image without modification
        else:
            frames = np.clip(255 * frames, 0, 255).astype(np.uint8)
            frames = np.transpose(frames, [2, 0, 1])
            return frames
    elif len(frames.shape) == 4:  # Video [T, H, W, C]
        # Sanity check that the channels dimension is last
        assert frames.shape[3] in [1, 3, 4], f"Invalid shape: {frames.shape}"
        is_depth = frames.shape[3] == 1
        frames = np.transpose(frames, [0, 3, 1, 2])
        # If the video is a float, convert it to uint8
        if np.issubdtype(frames.dtype, np.floating):
            if is_depth:
                frames = frames - frames.min()
                # Scale by 2 mean distances of near rays
                frames = frames / (2 * frames[frames <= 1].mean())
                # Scale to [0, 255]
                frames = np.clip(frames, 0, 1)
                # repeat channel dimension 3 times
                frames = np.repeat(frames, 3, axis=1)
            frames = np.clip(255 * frames, 0, 255).astype(np.uint8)
        return frames
    else:
        raise ValueError(f"Unexpected shape for {key}: {frames.shape}")

def evaluate_policy(policy, envs, device, config, log_video=False):
    """Evaluate a policy for a specified number of episodes.
    
    Args:
        policy: Policy to evaluate
        envs: Vectorized environment
        num_episodes: Number of episodes to evaluate
        device: Device to run evaluation on
        config: Configuration object
        log_video: Whether to log video frames
        
    Returns:
        dict of evaluation metrics
    """
    # Initialize metrics
    episode_returns = []
    episode_lengths = []
    num_episodes = config.eval.num_eval_episodes
    
    # Initialize video logging if enabled
    video_frames = {key: [] for key in envs.obs_space.keys() if key in ['image', 'camera_front']} if log_video else {}
    
    # Initialize observation storage
    obs = {}
    # Always use all keys from both teacher and student
    all_keys = set(
        policy.dual_encoder.student_encoder.mlp_keys + 
        policy.dual_encoder.student_encoder.cnn_keys +
        policy.dual_encoder.teacher_encoder.mlp_keys +
        policy.dual_encoder.teacher_encoder.cnn_keys
    )
    for key in all_keys:
        if len(envs.obs_space[key].shape) == 3 and envs.obs_space[key].shape[-1] == 3:  # Image observations
            obs[key] = torch.zeros((envs.num_envs,) + envs.obs_space[key].shape).to(device)
        else:  # Non-image observations
            size = np.prod(envs.obs_space[key].shape)
            obs[key] = torch.zeros((envs.num_envs, size)).to(device)
    
    # Initialize actions with zeros and reset flags
    action_shape = envs.act_space['action'].shape
    acts = {
        'action': np.zeros((envs.num_envs,) + action_shape, dtype=np.float32),
        'reset': np.ones(envs.num_envs, dtype=bool)
    }
    
    # Get initial observations
    obs_dict = envs.step(acts)
    next_obs = {}
    print("Available observation keys:", obs_dict.keys())
    print("Requested keys:", all_keys)
    for key in all_keys:
        if key not in obs_dict:
            print(f"Warning: Key '{key}' not found in observation dictionary")
            continue
        next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(device)
    next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)
    
    # Track episode returns and lengths for each environment
    env_returns = np.zeros(envs.num_envs)
    env_lengths = np.zeros(envs.num_envs)
    
    # Run evaluation until we have enough episodes
    while len(episode_returns) < num_episodes:
        # Get action from policy
        with torch.no_grad():
            action, _, _, _, _ = policy.get_action_and_value(obs)
        
        # Step environment
        action_np = action.cpu().numpy()
        if policy.is_discrete:
            action_np = action_np.reshape(envs.num_envs, -1)
        
        acts = {
            'action': action_np,
            'reset': next_done.cpu().numpy()
        }
        
        obs_dict = envs.step(acts)
        
        # Store video frames if logging
        if log_video:
            for key in video_frames.keys():
                if key in obs_dict:
                    video_frames[key].append(obs_dict[key][0].copy())
        
        # Process observations
        for key in all_keys:
            obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(device)
        next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)
        
        # Update episode tracking
        for env_idx in range(envs.num_envs):
            env_returns[env_idx] += obs_dict['reward'][env_idx]
            env_lengths[env_idx] += 1
            
            if obs_dict['is_last'][env_idx]:
                # Store episode metrics if we haven't collected enough episodes yet
                if len(episode_returns) < num_episodes:
                    episode_returns.append(env_returns[env_idx])
                    episode_lengths.append(env_lengths[env_idx])
                
                # Reset environment tracking
                env_returns[env_idx] = 0
                env_lengths[env_idx] = 0
    
    # Convert lists to numpy arrays for statistics
    episode_returns = np.array(episode_returns)
    episode_lengths = np.array(episode_lengths)
    
    metrics = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths)
    }
    
    if log_video:
        metrics['video_frames'] = video_frames
    
    return metrics

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parsed_args = parse_args()
    config = load_config(argv)

    # Seeding
    seed = parsed_args.seed if parsed_args.seed is not None else config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # Create environment
    # env = make_env(config, num_envs=config.num_envs)
    envs = make_envs(config, num_envs=config.num_envs)

    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    
    # Create shared dual encoder
    dual_encoder = DualEncoder(envs.obs_space, config).to(device)
    
    # Initialize teacher and student with shared encoder
    teacher = TeacherPolicy(envs, config, dual_encoder).to(device)
    student = StudentPolicy(envs, config, dual_encoder).to(device)
    
    # Debug prints for network structures
    print("\n" + "="*50)
    print("NEURAL NETWORK STRUCTURES")
    print("="*50)
    
    print("\nDUAL ENCODER STRUCTURE:")
    print("-"*30)
    print("Teacher Encoder:")
    print(dual_encoder.teacher_encoder)
    print("\nStudent Encoder:")
    print(dual_encoder.student_encoder)
    
    print("\nTEACHER POLICY STRUCTURE:")
    print("-"*30)
    print("Actor:")
    print(teacher.actor if teacher.is_discrete else teacher.actor_mean)
    print("\nCritic:")
    print(teacher.critic)
    
    print("\nSTUDENT POLICY STRUCTURE:")
    print("-"*30)
    print("Actor:")
    print(student.actor if student.is_discrete else student.actor_mean)
    print("\nCritic:")
    print(student.critic)
    
    print("\nPARAMETER COUNTS:")
    print("-"*30)
    print(f"Teacher Encoder: {sum(p.numel() for p in dual_encoder.teacher_encoder.parameters())}")
    print(f"Student Encoder: {sum(p.numel() for p in dual_encoder.student_encoder.parameters())}")
    print(f"Teacher Policy: {sum(p.numel() for p in teacher.parameters())}")
    print(f"Student Policy: {sum(p.numel() for p in student.parameters())}")
    print("="*50 + "\n")
        
    bc_trainer = BehavioralCloning(student, teacher, config)
    
    # Initialize optimizers
    teacher_optimizer = optim.Adam(teacher.parameters(), lr=config.learning_rate, eps=1e-5)
    student_optimizer = optim.Adam(student.parameters(), lr=config.bc.learning_rate, eps=1e-5)
    
    # Initialize logging
    exp_name = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{config.task}__{exp_name}__{seed}__{int(time.time())}"
    
    if config.track:
        import wandb
        wandb.init(
            project=config.wandb_project_name,
            entity=config.wandb_entity,
            sync_tensorboard=True,
            config=vars(config),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    
    # Training loop
    global_step = 0
    start_time = time.time()
    
    # Initialize video logging buffers
    teacher_video_frames = {key: [] for key in config.log_keys_video}
    student_video_frames = {key: [] for key in config.log_keys_video}
    last_video_log = 0
    video_log_interval = 10000  # Log a video every 10k steps
    
    # Initialize episode tracking buffers
    teacher_episode_returns = np.zeros(config.num_envs)
    teacher_episode_lengths = np.zeros(config.num_envs)
    student_episode_returns = np.zeros(config.num_envs)
    student_episode_lengths = np.zeros(config.num_envs)
    
    # Initialize observation storage
    obs = {}
    # Get union of teacher and student keys
    all_keys = set(teacher.mlp_keys + teacher.cnn_keys + student.mlp_keys + student.cnn_keys)
    for key in all_keys:
        if len(envs.obs_space[key].shape) == 3 and envs.obs_space[key].shape[-1] == 3:  # Image observations
            obs[key] = torch.zeros((config.num_steps, config.num_envs) + envs.obs_space[key].shape).to(device)
        else:  # Non-image observations
            size = np.prod(envs.obs_space[key].shape)
            obs[key] = torch.zeros((config.num_steps, config.num_envs, size)).to(device)
    
    # Initialize action storage with correct shape for one-hot if discrete
    if teacher.is_discrete:
        action_shape = (config.num_steps, config.num_envs, envs.act_space['action'].shape[0])
    else:
        action_shape = (config.num_steps, config.num_envs) + envs.act_space['action'].shape
    actions = torch.zeros(action_shape).to(device)
    logprobs = torch.zeros((config.num_steps, config.num_envs)).to(device)
    rewards = torch.zeros((config.num_steps, config.num_envs)).to(device)
    dones = torch.zeros((config.num_steps, config.num_envs)).to(device)
    values = torch.zeros((config.num_steps, config.num_envs)).to(device)
    
    # Initialize actions with zeros and reset flags
    action_shape = envs.act_space['action'].shape
    num_envs = config.num_envs
    
    acts = {
        'action': np.zeros((num_envs,) + action_shape, dtype=np.float32),
        'reset': np.ones(num_envs, dtype=bool)
    }
    
    # Get initial observations
    obs_dict = envs.step(acts)
    next_obs = {}
    for key in all_keys:
        if key not in obs_dict:
            print(f"Warning: Key '{key}' not found in observation dictionary")
            continue
        next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(device)
    next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)
    
    # Create evaluation environments
    eval_envs = make_envs(config, num_envs=config.eval.eval_envs)
    eval_envs.num_envs = config.eval.eval_envs
    
    # Initialize evaluation metrics
    last_eval = 0
    
    # Run initial evaluation
    print("Running initial evaluation...")
    teacher_eval_metrics = evaluate_policy(teacher, eval_envs, device, config, log_video=True)
    student_eval_metrics = evaluate_policy(student, eval_envs, device, config, log_video=True)
    
    # Log initial evaluation metrics
    if config.track:
        wandb.log({
            "eval_teacher/mean_return": teacher_eval_metrics['mean_return'],
            "eval_teacher/std_return": teacher_eval_metrics['std_return'],
            "eval_teacher/mean_length": teacher_eval_metrics['mean_length'],
            "eval_teacher/std_length": teacher_eval_metrics['std_length'],
            "eval_student/mean_return": student_eval_metrics['mean_return'],
            "eval_student/std_return": student_eval_metrics['std_return'],
            "eval_student/mean_length": student_eval_metrics['mean_length'],
            "eval_student/std_length": student_eval_metrics['std_length'],
            "metrics/global_step": 0,
        })
        
        # Log videos
        for key in teacher_eval_metrics['video_frames'].keys():
            if teacher_eval_metrics['video_frames'][key]:
                frames = np.stack(teacher_eval_metrics['video_frames'][key])
                processed_frames = process_video_frames(frames, key)
                wandb.log({
                    f"videos/teacher_{key}": wandb.Video(
                        processed_frames,
                        fps=10,
                        format="gif"
                    )
                }, step=0)
                
                if student_eval_metrics['video_frames'][key]:
                    frames = np.stack(student_eval_metrics['video_frames'][key])
                    processed_frames = process_video_frames(frames, key)
                    wandb.log({
                        f"videos/student_{key}": wandb.Video(
                            processed_frames,
                            fps=10,
                            format="gif"
                        )
                    }, step=0)
    
    # Main training loop
    num_iterations = config.total_timesteps // (config.num_envs * config.num_steps)
    for iteration in range(1, num_iterations + 1):
        if config.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * config.learning_rate
            teacher_optimizer.param_groups[0]["lr"] = lrnow
            student_optimizer.param_groups[0]["lr"] = lrnow
        
        # 1. Collect experience using teacher policy
        for step in range(0, config.num_steps):
            global_step += config.num_envs
            
            # Store observations
            for key in all_keys:
                obs[key][step] = next_obs[key]
            dones[step] = next_done
            
            # Get action from teacher policy
            with torch.no_grad():
                action, logprob, _, value, _ = teacher.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            
            # Step environment
            action_np = action.cpu().numpy()
            # print(f"Raw action shape (training): {action_np.shape}")
            if teacher.is_discrete:
                action_np = action_np.reshape(config.num_envs, -1)
                # print(f"Final action shape: {action_np.shape}")
            
            acts = {
                'action': action_np,
                'reset': next_done.cpu().numpy()
            }
            
            obs_dict = envs.step(acts)
            
            # Store raw observations for video logging before any processing
            for key in config.log_keys_video:
                if key in obs_dict:
                    teacher_video_frames[key].append(obs_dict[key][0].copy())
            
            # Process observations
            for key in all_keys:
                next_obs[key] = torch.Tensor(obs_dict[key].astype(np.float32)).to(device)
            next_done = torch.Tensor(obs_dict['is_last'].astype(np.float32)).to(device)
            rewards[step] = torch.tensor(obs_dict['reward'].astype(np.float32)).to(device).view(-1)
            
            # Store transition in replay buffer
            for env_idx in range(config.num_envs):
                transition = {
                    'obs': {key: obs[key][step, env_idx].cpu().numpy() for key in all_keys},
                    'action': actions[step, env_idx].cpu().numpy(),
                    'reward': rewards[step, env_idx].cpu().numpy(),
                    'next_obs': {key: next_obs[key][env_idx].cpu().numpy() for key in all_keys},
                    'done': next_done[env_idx].cpu().numpy()
                }
            
            # Update episode tracking
            for env_idx in range(config.num_envs):
                teacher_episode_returns[env_idx] += obs_dict['reward'][env_idx]
                teacher_episode_lengths[env_idx] += 1
                
                if obs_dict['is_last'][env_idx]:
                    print(f"global_step={global_step}, teacher_episode_return={teacher_episode_returns[env_idx]}, teacher_episode_length={teacher_episode_lengths[env_idx]}")
                    
                    if config.track:
                        wandb.log({
                            "charts/teacher_episode_return": teacher_episode_returns[env_idx],
                            "charts/teacher_episode_length": teacher_episode_lengths[env_idx],
                            "metrics/global_step": global_step,
                        })
                    
                    # Reset episode tracking
                    teacher_episode_returns[env_idx] = 0
                    teacher_episode_lengths[env_idx] = 0
        
        # 2. Compute advantages for PPO
        with torch.no_grad():
            next_value = teacher.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        
        # 3. PPO update for teacher
        batch_size = int(config.num_envs * config.num_steps)
        minibatch_size = int(batch_size // config.num_minibatches)
        b_inds = np.arange(batch_size)
        clipfracs = []
        
        for epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                # Create minibatch observations dictionary
                mb_obs = {}
                for key in all_keys:
                    mb_obs[key] = obs[key].reshape(-1, *obs[key].shape[2:])[mb_inds]
                
                # Get new action probabilities and values
                _, newlogprob, entropy, newvalue, imitation_losses = teacher.get_action_and_value(
                    mb_obs, actions.reshape(-1, *actions.shape[2:])[mb_inds]
                )
                
                # Calculate PPO loss
                logratio = newlogprob - logprobs.reshape(-1)[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]
                
                mb_advantages = advantages.reshape(-1)[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - returns.reshape(-1)[mb_inds]) ** 2
                    v_clipped = values.reshape(-1)[mb_inds] + torch.clamp(
                        newvalue - values.reshape(-1)[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns.reshape(-1)[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns.reshape(-1)[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Imitation loss
                imitation_loss = 0.0
                if imitation_losses and config.encoder.teacher_to_student_imitation:
                    imitation_loss = imitation_losses['teacher_to_student']
                                
                # Total loss
                loss = (
                    pg_loss 
                    - config.ent_coef * entropy_loss 
                    + v_loss * config.vf_coef 
                    + config.encoder.teacher_to_student_lambda * imitation_loss
                )
                
                # Optimize teacher
                teacher_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(teacher.parameters(), config.max_grad_norm)
                teacher_optimizer.step()
            
            # Early stopping
            target_kl = config.target_kl
            if isinstance(target_kl, str) and target_kl.lower() == 'none':
                target_kl = None
            if target_kl is not None and approx_kl > float(target_kl):
                break
        
        # 4. Student learns from teacher via BC
        # First collect student trajectories - same number of steps as teacher
        student_transitions = student.collect_transitions(envs, config.num_steps)
        
        # Then do BC updates using the collected data
        for _ in range(config.update_epochs * config.num_minibatches):  # 4 epochs * 4 minibatches
            bc_metrics = bc_trainer.train(
                student_transitions,
                num_epochs=1
            )
            
            # Log BC metrics
            if config.track:
                wandb.log({
                    "bc/loss": bc_metrics['bc_loss'],
                    "bc/action_diff": bc_metrics.get('action_diff', None),
                    "metrics/global_step": global_step,
                })
        
        # Reset episode tracking for student
        student_episode_returns = np.zeros(config.num_envs)
        student_episode_lengths = np.zeros(config.num_envs)
        
        # Process student transitions (only for training, no logging)
        for transition in student_transitions:
            for env_idx in range(config.num_envs):
                if transition['done']:
                    # Reset episode tracking
                    student_episode_returns[env_idx] = 0
                    student_episode_lengths[env_idx] = 0
                else:
                    # Update episode tracking
                    student_episode_returns[env_idx] += transition['reward']
                    student_episode_lengths[env_idx] += 1
        
        # Periodic evaluation
        if global_step - last_eval >= config.eval.eval_interval * config.num_envs:
            # Evaluate teacher
            teacher_eval_metrics = evaluate_policy(teacher, eval_envs, device, config, log_video=True)
            
            # Evaluate student
            student_eval_metrics = evaluate_policy(student, eval_envs, device, config, log_video=True)
            
            # Log evaluation metrics
            if config.track:
                # Log metrics
                wandb.log({
                    "eval_teacher/mean_return": teacher_eval_metrics['mean_return'],
                    "eval_teacher/std_return": teacher_eval_metrics['std_return'],
                    "eval_teacher/mean_length": teacher_eval_metrics['mean_length'],
                    "eval_teacher/std_length": teacher_eval_metrics['std_length'],
                    "eval_student/mean_return": student_eval_metrics['mean_return'],
                    "eval_student/std_return": student_eval_metrics['std_return'],
                    "eval_student/mean_length": student_eval_metrics['mean_length'],
                    "eval_student/std_length": student_eval_metrics['std_length'],
                    "metrics/global_step": global_step,
                })
                
                # Log videos every video_log_interval evaluations
                eval_count = global_step // (config.eval.eval_interval * config.num_envs)
                if eval_count % config.eval.video_log_interval == 0:
                    # Teacher video
                    if 'image' in teacher_eval_metrics['video_frames'] and teacher_eval_metrics['video_frames']['image']:
                        frames = np.stack(teacher_eval_metrics['video_frames']['image'])
                        processed_frames = process_video_frames(frames, 'image')
                        wandb.log({
                            "videos/teacher_image": wandb.Video(
                                processed_frames,
                                fps=10,
                                format="gif"
                            )
                        }, step=global_step)
                    
                    if 'heatmap' in teacher_eval_metrics['video_frames'] and teacher_eval_metrics['video_frames']['heatmap']:
                        frames = np.stack(teacher_eval_metrics['video_frames']['heatmap'])
                        processed_frames = process_video_frames(frames, 'heatmap')
                        wandb.log({
                            "videos/teacher_heatmap": wandb.Video(
                                processed_frames,
                                fps=10,
                                format="gif"
                            )
                        }, step=global_step)
                    
                    # Student video
                    if 'image' in student_eval_metrics['video_frames'] and student_eval_metrics['video_frames']['image']:
                        frames = np.stack(student_eval_metrics['video_frames']['image'])
                        processed_frames = process_video_frames(frames, 'image')
                        wandb.log({
                            "videos/student_image": wandb.Video(
                                processed_frames,
                                fps=10,
                                format="gif"
                            )
                        }, step=global_step)
                    
                    if 'heatmap' in student_eval_metrics['video_frames'] and student_eval_metrics['video_frames']['heatmap']:
                        frames = np.stack(student_eval_metrics['video_frames']['heatmap'])
                        processed_frames = process_video_frames(frames, 'heatmap')
                        wandb.log({
                            "videos/student_heatmap": wandb.Video(
                                processed_frames,
                                fps=10,
                                format="gif"
                            )
                        }, step=global_step)
            
            last_eval = global_step
        
        # Log metrics
        if config.track:
            wandb.log({
                "charts/learning_rate": teacher_optimizer.param_groups[0]["lr"],
                "losses/policy_loss": pg_loss.item(),
                "losses/value_loss": v_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
                "charts/SPS": int(global_step / (time.time() - start_time)),
                "metrics/global_step": global_step,
            })
    
    if config.save_model:
        model_path = f"runs/{run_name}/{exp_name}.cleanrl_model"
        torch.save({
            'teacher_state_dict': teacher.state_dict(),
            'student_state_dict': student.state_dict(),
        }, model_path)
        print(f"model saved to {model_path}")

    envs.close()

if __name__ == "__main__":
    main()
