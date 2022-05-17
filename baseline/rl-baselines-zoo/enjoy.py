import argparse
import os
import warnings
import sys
import pkg_resources
import importlib
import stable_baselines
# plot
import matplotlib.pyplot as plt

import scipy.stats as ss
import scipy

# For pybullet envs
warnings.filterwarnings("ignore")
import gym
try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None
import numpy as np
try:
    import highway_env
except ImportError:
    highway_env = None

from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, VecEnv

from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams

# Fix for breaking change in v2.6.0
#if pkg_resources.get_distribution("stable_baselines").version >= "2.6.0":
sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.deepq.replay_buffer
stable_baselines.deepq.replay_buffer.Memory = stable_baselines.deepq.replay_buffer.ReplayBuffer

def kl_scipy(p, q):
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    p = p.flatten()
    q = q.flatten()
    p[p==0] = np.finfo(float).eps
    q[q==0] = np.finfo(float).eps
    if(len(p)>1):
        pg = ss.gaussian_kde(p)
        qg = ss.gaussian_kde(q)
        kl = ss.entropy(pg(p),qg(q))
        print("p,q",ss.entropy(pg(p),qg(q)))
        print("len of p",len(p))
        return kl
    else:
        return 0
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1')
    parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
    parser.add_argument('-fq', '--folder_q', help='Log folder', type=str, default='trained_agents')
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=1000,
                        type=int)
    parser.add_argument('--n-envs', help='number of environments', default=1,
                        type=int)
    parser.add_argument('--exp-id', help='Experiment ID (default: -1, no exp folder, 0: latest)', default=-1,
                        type=int)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--no-render', action='store_true', default=False,
                        help='Do not render the environment (useful for tests)')
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Use deterministic actions')
    parser.add_argument('--stochastic', action='store_true', default=False,
                        help='Use stochastic actions (for DDPG/DQN/SAC)')
    parser.add_argument('--norm-reward', action='store_true', default=False,
                        help='Normalize reward if applicable (trained with VecNormalize)')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--reward-log', help='Where to log reward', default='', type=str)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[], help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.folder
    folder_q = args.folder_q

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print('Loading latest experiment, id={}'.format(args.exp_id))

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, '{}_{}'.format(env_id, args.exp_id))
    else:
        log_path = os.path.join(folder, algo)

    # Sanity checks
    if args.exp_id > 0:
        log_path_q = os.path.join(folder_q, algo, '{}_{}'.format(env_id, args.exp_id))
    else:
        log_path_q = os.path.join(folder_q, algo)

    assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)

    found = False
    for ext in ['pkl', 'zip']:
        model_path = "{}/{}.{}".format(log_path, env_id, ext)
        found = os.path.isfile(model_path)
        if found:
            break
    found_q = False
    for ext in ['pkl', 'zip']:
        model_path_q = "{}/{}.{}".format(log_path_q, env_id, ext)
        found_q = os.path.isfile(model_path_q)
        if found_q:
            break

    if not found:
        raise ValueError("No model found for {} on {}, path: {}".format(algo, env_id, model_path))

    if algo in ['dqn', 'ddpg', 'sac', 'td3']:
        args.n_envs = 1

    set_global_seeds(args.seed)

    is_atari = 'NoFrameskip' in env_id

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=args.norm_reward, test_mode=True)

    log_dir = args.reward_log if args.reward_log != '' else None

    env = create_test_env(env_id, n_envs=args.n_envs, is_atari=is_atari,
                          stats_path=stats_path, seed=args.seed, log_dir=log_dir,
                          should_render=not args.no_render,
                          hyperparams=hyperparams)

    # ACER raises errors because the environment passed must have
    # the same number of environments as the model was trained on.
    load_env = None if algo == 'acer' else env
    model = ALGOS[algo].load(model_path, env=load_env)

    model_q = ALGOS[algo].load(model_path_q, env=load_env)

    obs = env.reset()

    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.deterministic or algo in ['dqn', 'ddpg', 'sac', 'her', 'td3'] and not args.stochastic

    episode_reward = 0.0
    episode_rewards = []
    ep_len = 0
    # For HER, monitor success rate
    successes = []
    action_q_list = []
    action_list = []
    kl_list = []
    for _ in range(args.n_timesteps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action_q, _ = model_q.predict(obs, deterministic=deterministic)
        action_list.append(action.flatten().tolist())
        action_q_list.append(action_q.flatten().tolist())

        # Random Agent
        # action = [env.action_space.sample()]
        # Clip Action to avoid out of bound errors
        if isinstance(env.action_space, gym.spaces.Box):
            action = np.clip(action, env.action_space.low, env.action_space.high)
        obs, reward, done, infos = env.step(action)
        if not args.no_render:
            env.render('human')

        episode_reward += reward[0]
        ep_len += 1

        if args.n_envs == 1:
            # For atari the return reward is not the atari score
            # so we have to get it from the infos dict
            if is_atari and infos is not None and args.verbose >= 1:
                episode_infos = infos[0].get('episode')
                if episode_infos is not None:
                    print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
                    print("Atari Episode Length", episode_infos['l'])

                    # calculate KL-divergence
                    flat_action = [item for sublist in action_list for item in sublist]
                    flat_action_q = [item for sublist in action_q_list for item in sublist]
                    kl_list.append(kl_scipy(flat_action, flat_action_q))
                    plt.hist(flat_action, bins=20, label='action')
                    plt.hist(flat_action_q, bins=20, label='action_q')
                    plt.legend()
                    # save the figure
                    # append the env-name to the file-name
                    # appen algo name to the file-name
                    plt.savefig(os.path.join('action_hist_' + env_id + '_' + algo + '.png'))
                    plt.close()
                    flat_action = []
                    flat_action_q = []
                    action_list = []
                    action_q_list = []

            if done and not is_atari and args.verbose > 0:
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                print("Episode Reward: {:.2f}".format(episode_reward))
                print("Episode Length", ep_len)
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                ep_len = 0

                # calculate KL-divergence
                flat_action = [item for sublist in action_list for item in sublist]
                flat_action_q = [item for sublist in action_q_list for item in sublist]
                kl_list.append(kl_scipy(flat_action, flat_action_q))
                plt.hist(flat_action, bins=20, label='action')
                plt.hist(flat_action_q, bins=20, label='action_q')
                plt.legend()
                # save the figure
                # append the env-name to the file-name
                plt.savefig(os.path.join('action_hist_' + env_id + '_' + algo + '.png'))
                plt.close()
                flat_action = []
                flat_action_q = []
                action_list = []
                action_q_list = []

            # Reset also when the goal is achieved when using HER
            if done or infos[0].get('is_success', False):
                if args.algo == 'her' and args.verbose > 1:
                    print("Success?", infos[0].get('is_success', False))
                # Alternatively, you can add a check to wait for the end of the episode
                # if done:
                obs = env.reset()
                if args.algo == 'her':
                    successes.append(infos[0].get('is_success', False))
                    episode_reward, ep_len = 0.0, 0

    if args.verbose > 0 and len(successes) > 0:
        print("Success rate: {:.2f}%".format(100 * np.mean(successes)))

    if args.verbose > 0 and len(episode_rewards) > 0:
        print("Mean reward: {:.2f}".format(np.mean(episode_rewards)))

    # Workaround for https://github.com/openai/gym/issues/893
    if not args.no_render:
        if args.n_envs == 1 and 'Bullet' not in env_id and not is_atari and isinstance(env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecNormalize) or isinstance(env, VecFrameStack):
                env = env.venv
            env.envs[0].env.close()
        else:
            # SubprocVecEnv
            env.close()

    # plot histogram of action and action_q


    #plt.savefig('action_histogram.png')

    # calculate kl-divergence over action dist
    # get the mean of a list
    print("KL-Lists:", kl_list)
    mean_kl = np.mean(kl_list)
    print("Mean KL-Divergence: {:.5f}".format(mean_kl))

if __name__ == '__main__':
    main()
