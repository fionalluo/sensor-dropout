import re

import embodied
import numpy as np
import numbers


def train_full_eval_drivers(
    agent, train_env, full_train_env, eval_env, full_eval_env, train_replay, full_train_replay, eval_replay, full_eval_replay, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  print('Logdir', logdir)
  should_expl = embodied.when.Until(args.expl_until)
  should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
  should_sync = embodied.when.Every(args.sync_every)
  should_rollout = embodied.when.Every(args.policy_rollout_every)
  should_full_rollout = embodied.when.Every(args.full_policy_rollout_every)
  
  step = logger.step
  updates = embodied.Counter()
  metrics = embodied.Metrics()
  print('Observation space:', embodied.format(train_env.obs_space), sep='\n')
  print('Action space:', embodied.format(train_env.act_space), sep='\n')

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
  timer.wrap('env', train_env, ['step'])
  if hasattr(train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])
  if hasattr(full_train_replay, '_sample'):
    timer.wrap('replay', train_replay, ['_sample'])

  nonzeros = set()
  def per_episode(ep, mode):
    """Method called per episode. Ep is a single episode dictionary, mode is a string for the mode."""
    length = len(ep['reward']) - 1  # length of episode
    score = float(ep['reward'].astype(np.float64).sum())  # score = sum of rewards
    logger.add({  # log the...
        'length': length,  # length of episode
        'score': score,  # total score
        'reward_rate': (ep['reward'] - ep['reward'].min() >= 0.1).mean(),  # reward rate (prop. states with reward)
    }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))  # episode if train, else eval_episode, full_eval_episode
    print(f'Episode has {length} steps and return {score:.2f}.')  # print length and score for debugging
    stats = {}  # dictionary for stats
    for key in args.log_keys_video:  # for every key in log_keys_video
      if key in ep:  # if the key is in the episode
        stats[f'policy_{key}'] = ep[key]  # add key to stats dictionary
    for key, value in ep.items():  # for every key, value in episode
      if not args.log_zeros and key not in nonzeros and (value == 0).all():  # skip zero values
        continue
      nonzeros.add(key)  # add key to nonzeros set of key names
      if re.match(args.log_keys_sum, key):  # if key matches log_keys_sum
        stats[f'sum_{key}'] = ep[key].sum()  # add sum of key to stats
      if re.match(args.log_keys_mean, key):  # if key matches log_keys_mean
        stats[f'mean_{key}'] = ep[key].mean()  # add mean of key to stats
      if re.match(args.log_keys_max, key):  # if key matches log_keys_max
        stats[f'max_{key}'] = ep[key].max(0).mean()  # add max of key to stats
    metrics.add(stats, prefix=f'{mode}_stats')  # add stats dict to all metrics with prefix of mode_stats
  
  def per_call(call_stats, mode):
    """Method called on call. Mode is a string for the mode."""
    # If environment is Path Bandit Env, add a metric for which goal the agent reached
    if args.task_name == "bandit":
      # print("LAST_REWARD", call_stats["last_reward"])
      call_stats["optimal"] = (np.abs(np.array(call_stats["last_reward"]) - (1.0)) <= 0.01).astype(int)
      call_stats["suboptimal"] = (np.abs(np.array(call_stats["last_reward"]) - (0.3)) <= 0.01).astype(int)
      call_stats["penalty"] = (np.abs(np.array(call_stats["last_reward"]) - (-1.0)) <= 0.01).astype(int)
      call_stats["unfinished"] = (np.abs(np.array(call_stats["last_reward"]) - (0.01)) <= 0.01).astype(int)

    for key, value_list in call_stats.items():
      if len(value_list) == 0:
        continue
      if not isinstance(value_list[0], numbers.Number):
        # if it's a number or a high-dimensional output, skip for now
        continue
      value = np.stack(value_list)
      logger.add({
          f'{key}_mean': value.mean(),
          f'{key}_max': value.max(),
          f'{key}_min': value.min(),
      }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))
    
    # add the heatmap if it exists in full_eval_env
    if len(call_stats["heatmap"]) > 0:
      logger.add({
          'heatmap': call_stats["heatmap"][-1]
      }, prefix=('episode' if mode == 'train' else f'{mode}_episode'))

  driver_train = embodied.Driver(train_env)
  driver_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  driver_train.on_step(lambda tran, _: step.increment())
  driver_train.on_step(train_replay.add)  # default

  driver_full_train = embodied.Driver(full_train_env)
  driver_full_train.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
  driver_full_train.on_step(lambda tran, _: step.increment())
  driver_full_train.on_step(full_train_replay.add)  # default

  driver_eval = embodied.Driver(eval_env)
  driver_eval.on_step(eval_replay.add)
  driver_eval.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))
  driver_eval.on_call(lambda call_stats: per_call(call_stats, mode='eval'))

  driver_full_eval = embodied.Driver(full_eval_env)
  driver_full_eval.on_step(full_eval_replay.add)  # default / original
  driver_full_eval.on_episode(lambda ep, worker: per_episode(ep, mode='full_eval'))
  driver_full_eval.on_call(lambda call_stats: per_call(call_stats, mode='full_eval'))

  random_agent = embodied.RandomAgent(train_env.act_space)
  print('Prefill train dataset.')
  while len(train_replay) < max(args.batch_steps, args.train_fill):
    driver_train(random_agent.policy, steps=100)
  print('Prefill full train dataset.')
  while len(full_train_replay) < max(args.batch_steps, args.train_fill):
    driver_full_train(random_agent.policy, steps=100)
  print('Prefill eval dataset.')
  while len(eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_eval(random_agent.policy, steps=100)
  print('Prefill full_eval dataset.')
  while len(full_eval_replay) < max(args.batch_steps, args.eval_fill):
    driver_full_eval(random_agent.policy, steps=100)
  logger.add(metrics.result())
  logger.write()

  dataset_train = agent.dataset(train_replay.dataset)
  dataset_eval = agent.dataset(eval_replay.dataset)
  state = [None]  # To be writable from train step function below.
  batch = [None]
  def train_step(tran, worker):
    # Train step is called on every on_step of driver_train.
    for _ in range(should_train(step)):
      with timer.scope('dataset_train'):
        batch[0] = next(dataset_train)  # get batch from train_replay
      outs, state[0], mets = agent.train(batch[0], state[0])
      metrics.add(mets, prefix='train')
      if 'priority' in outs:  # prioritize certain batches - not really needed
        train_replay.prioritize(outs['key'], outs['priority'])
      updates.increment()
    if should_sync(updates):
      agent.sync()
    if should_log(step):
      logger.add(metrics.result())
      logger.add(agent.report(batch[0]), prefix='report')
      with timer.scope('dataset_eval'):
        eval_batch = next(dataset_eval)
      logger.add(agent.report(eval_batch), prefix='eval')
      logger.add(train_replay.stats, prefix='replay')
      logger.add(eval_replay.stats, prefix='eval_replay')
      logger.add(timer.stats(), prefix='timer')
      logger.write(fps=True)
  driver_train.on_step(train_step)
  driver_full_train.on_step(train_step)

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we jused saved.

  print('Start training loop.')
  policy_train = lambda *args: agent.policy(
      *args, mode='explore' if should_expl(step) else 'train')
  full_policy_train = lambda *args: agent.policy(
    *args, mode='explore' if should_expl(step) else 'full_train')
  policy_eval = lambda *args: agent.policy(*args, mode='eval')
  full_policy_eval = lambda *args: agent.policy(*args, mode='full_eval')
  algo_updates = embodied.Counter()

  while step < args.steps:
    if should_eval(step):
      print('Starting evaluation at step', int(step))
      driver_eval.reset()
      driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps))
      driver_full_eval.reset()
      driver_full_eval(full_policy_eval, episodes=max(len(full_eval_env), args.eval_eps))

    if should_full_rollout(algo_updates):
      driver_full_train(full_policy_train, episodes=1)
    if should_rollout(algo_updates):
      driver_train(policy_train, episodes=1)
    if should_save(step):
      checkpoint.save()
    algo_updates.increment()
  logger.write()
