import csv
import tensorflow as tf
import argparse
import numpy as np
import gym
import os
import pybullet_envs
import subprocess
import sys
import pkg_resources
import stable_baselines

from stable_baselines import bench, logger
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.common import tf_util, set_global_seeds

if __name__=="__main__":

    if pkg_resources.get_distribution("stable_baselines").version >= "2.6.0":
        sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.deepq.replay_buffer
        stable_baselines.deepq.replay_buffer.Memory = stable_baselines.deepq.replay_buffer.ReplayBuffer

    sys.path.insert(0, '../rl-baselines-zoo/')
    file_dir = os.path.dirname(os.path.abspath(__file__))
    quarl_directory = "/".join(file_dir.split("/")[:-1])
    from utils import create_test_env, get_saved_hyperparams, ALGOS

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="CartPole-v1", help='environment ID(s)')
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                            type=str, required=False, choices=list(ALGOS.keys()))
    #parser.add_argument('-q', '--quant-bit-width', help='Set bit width for quantization', default=8, type=int)
    #parser.add_argument('--base', help="Set base directory for saved models", default="/home/vj-reddi/quantization-air-learning/", type=str)
    parser.add_argument('--fp16', help="Run Float16 Post Training quantization", type=int, default=0)
    parser.add_argument('--fp32', help="Run Float32 TFLite model", type=int, default=0)
    parser.add_argument('--input-node', help="Input nodes for post training quantization", type=str, default="input/Ob")
    parser.add_argument('--int', help="Run Optimize for size tflite", default=0, type=int)
    parser.add_argument('--output-node', help='Output nodes for post training quantization', type=str, default="output/Softmax")
    parser.add_argument('-n', help='Set number of iterations to evaluate model on', type=int, default=100)
    args = parser.parse_args()

    class Algo(ALGOS[args.algo]):
        def __init__(self, policy, env, **kwargs):
            super(Algo, self).__init__(policy=policy, env=env, **kwargs)

        def save_graph(self, dir, name):
            with self.graph.as_default():
                saver = tf.train.Saver()
                tf.train.write_graph(self.sess.graph_def, dir, name + '.pb')
                saver.save(self.sess, dir + '/' + name + '.ckpt')

    models_path = os.path.join(quarl_directory, "rl-baselines-zoo/trained_agents")
    os.makedirs(models_path, exist_ok=True)
    ckpt_path = os.path.join(quarl_directory,"saved/")
    os.makedirs(ckpt_path, exist_ok=True)
    frozen_path = os.path.join(quarl_directory, "frozen/")
    quant_path = os.path.join(quarl_directory,"quant/")
    tflite_path = "/tmp/foo.tflite"
    if args.int:
        type = "8"
        ckpt_model_path = os.path.join(ckpt_path, "8", args.algo)
        frozen_model_path = os.path.join(frozen_path, "8", args.algo, args.env + ".pb")
    if args.fp16:
        type = "16"
        ckpt_model_path = os.path.join(ckpt_path, "16", args.algo)
        frozen_model_path = os.path.join(frozen_path, "16", args.algo, args.env + ".pb")
    elif args.fp32:
        type = "32"
        ckpt_model_path = os.path.join(ckpt_path, "32", args.algo)
        frozen_model_path = os.path.join(frozen_path, "32", args.algo, args.env + ".pb")
    os.makedirs(os.path.join(frozen_path,type,args.algo), exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(quant_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(ckpt_model_path, exist_ok=True)

    is_atari = 'NoFrameskip' in args.env
    is_bullet = "BulletEnv" in args.env
    if_cont = "Continuous" in args.env
    #print(args.base)
    if is_atari:
        input_nodes = {"dqn":"deepq/input/Cast", "ppo2":"input/Cast", "a2c":"input/Cast", "ddpg":"input/input/Ob"}
        output_nodes = {"dqn":"deepq/model/add", "ppo2":"model/pi/add", "a2c":"model/pi/add", "ddpg":"model/pi/Tanh"}
    elif is_bullet:
        input_nodes = {"ppo2":"input/Cast", "a2c":"input/Cast", "ddpg":"input/input/Ob"}
        output_nodes = {"ppo2":"model/pi/add", "a2c":"model/pi/add", "ddpg":"model/pi/Tanh"}
    else:
        if if_cont:
            input_nodes = {"ppo2":"input/Ob", "a2c":"input/Ob", "ddpg":"input/input/Ob"}
            output_nodes = {"ppo2":"model/pi/add", "a2c":"model/pi/add", "ddpg":"model/pi/Tanh"}
        else:
            input_nodes = {"dqn":"deepq/input/Ob", "ppo2":"input/Ob", "a2c":"input/Ob", "ddpg":"input/input/Ob"}
            output_nodes = {"dqn":"deepq/model/add", "ppo2":"model/pi/add", "a2c":"model/pi/add", "ddpg":"model/pi/Tanh"}

    model_path = os.path.join(models_path, args.algo, args.env + ".pkl")
    print("Loading model from", model_path)

    input_node, output_node = input_nodes[args.algo], output_nodes[args.algo]
    og_model = Algo.load(model_path)
    og_model.save_graph(ckpt_model_path, args.env)
    subprocess.call(['./freeze.sh', args.algo, args.env, output_node, type, quarl_directory])
    converter = tf.lite.TFLiteConverter.from_frozen_graph(frozen_model_path, [input_node], [output_node])

    if args.int:
        print("Int 8 weight quantization")
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    elif args.fp16:
        print("Fp16 quantization")
        converter.optimizations=[tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
    elif args.fp32:
        print("Fp32 TFLite")
    tflite_model = converter.convert()
    tflite_model_path = "/tmp/foo.tflite"
    open(tflite_model_path, 'wb').write(tflite_model)

    tflite_model_path = "/tmp/foo.tflite"
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    log_dir = "/tmp"
    algo_path = os.path.join(models_path, args.algo)
    hyperparams, stats_path = get_saved_hyperparams(model_path, norm_reward=False, test_mode=True)

    print("Running ", args.algo, " on ", args.env)
    set_global_seeds(0)
    env = create_test_env(args.env, n_envs=1, is_atari=is_atari,
                              stats_path=stats_path, seed=0, log_dir=log_dir,
                              should_render=False,
                              hyperparams=hyperparams)

    print("Evaluating converted model")
    num_steps = args.n
    episode_rewards, lengths, norm_rewards = [], [], []
    for i in range(num_steps):
        obs, done = env.reset(), False
        episode_rew, norm_rew, length = 0, 0.0, 0
        while not done:
            if is_atari:
                obs = obs.astype(np.float32)
            interpreter.set_tensor(input_index, obs)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_index)
            action = predictions
            if isinstance(env.action_space, gym.spaces.Box) and args.algo!="ddpg":
                mean, std = np.split(predictions, 2, axis=len(predictions.shape) - 1)
                std = np.exp(std)
                action = mean + std*np.random.normal(size = mean.shape)
            if args.algo=="ddpg":
                action = predictions
            if args.env == "CartPole-v1" or args.env == "MountainCar-v0" or args.algo == "dqn":
                action = np.argmax(predictions, axis=1)
            if is_atari and args.algo != "dqn":
                action = np.argmax(predictions - np.log(-1*np.log(np.random.uniform(size=predictions.shape))) , axis=1)
            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, rewards, done, infos = env.step(action)
            episode_infos = infos[0].get('episode')
            if episode_infos is not None:
                print("Episode score: {:.2f}".format(episode_infos['r']))
                print("Episode length", episode_infos['l'])
                episode_rew += episode_infos['r']
                length += episode_infos['l']
                if done:
                    episode_rewards.append(episode_rew)
                    lengths.append(length)
                    norm_rew = np.mean(episode_rew)/np.std(episode_rew)
                    norm_rewards.append(norm_rew)
            if done or infos[0].get('is_success', False):
                done = True
    print("Average Reward ", np.mean(episode_rewards))

    directory = os.path.join(quarl_directory, "csvs", type, args.algo, args.env)
    os.makedirs(directory, exist_ok=True)
    with open(directory + "rewards.csv", 'w') as file:
        writer = csv.writer(file, delimiter=',',  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(episode_rewards)

