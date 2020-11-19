import os
import csv
import sys
import gym
import numpy as np
import subprocess
import argparse
import tensorflow as tf

from stable_baselines import bench, logger
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.common import tf_util, set_global_seeds

sys.path.insert(0, '../rl-baselines-zoo/')
file_dir = os.path.dirname(os.path.abspath(__file__))
quarl_directory = "/".join(file_dir.split("/")[:-1])
from utils import create_test_env, get_saved_hyperparams, ALGOS

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="CartPole-v1", help='environment ID(s)')
parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                        type=str, required=False)
parser.add_argument('-q', '--quant-bit-width', help='Set bit width for quantization', default=8, type=int)
parser.add_argument('--base', help="Set base directory for saved models", default="/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1]), type=str)
parser.add_argument('-n', help='Set number of iterations to evaluate model on', type=int, default=10000000)
parser.add_argument('--quant-delay', help="Set quant delay", type=int, default=5000000)
args = parser.parse_args()

is_atari = "NoFrameskip" in args.env
is_bullet = "BulletEnv" in args.env
if_cont = "Continuous" in args.env
print(args.base)
if is_atari:
    input_nodes = {"dqn":"deepq/input/Ob", "ppo2":"input/Ob", "a2c":"input/Ob", "acer":"input/sub", "ddpg":"input/input/Ob", "acktr":"input/sub"}
    output_nodes = {"dqn":"deepq/model/add", "ppo2":"model/pi/add", "a2c":"model/pi/add", "acer":"output/ArgMax", "ddpg":"model/pi/Tanh", "acktr":"output/Argmax_1"}
elif is_bullet:
    input_nodes = {"dqn":None, "ppo2":"input/Ob", "a2c":"input/Ob", "ddpg":"input/input/Ob"}
    output_nodes = {"dqn":None, "ppo2":"model/pi/add", "a2c":"model/pi/add", "ddpg":"model/pi/Tanh", "acer":"output/ArgMax"}
else:
    if if_cont:
        input_nodes = {"dqn":"deepq/input/Ob", "ppo2":"input/Ob", "a2c":"input/Ob", "ddpg":"input/input/Ob", "acer":"input/sub", "acktr":"input/Ob"}
        output_nodes = {"dqn":"deepq/model/add", "ppo2":"model/concat", "a2c":"model/concat", "ddpg":"model/pi/Tanh", "acer":"output/add", "acktr":"output/Argmax"}
    else:
        input_nodes = {"dqn":"deepq/input/Ob", "ppo2":"input/Ob", "a2c":"input/Ob", "ddpg":"input/input/Ob", "acer":"input/sub", "acktr":"input/Ob"}
        output_nodes = {"dqn":"deepq/model/add", "ppo2":"model/pi/add", "a2c":"model/pi/add", "ddpg":"model/pi/Tanh", "acer":"output/add", "acktr":"output/Argmax"}                                                                                                                        
input_node, output_node = input_nodes[args.algo], output_nodes[args.algo]
print(args.base)
train_graph_dir = os.path.join(args.base, "quant_train/train", str(args.quant_bit_width), args.algo)
os.makedirs(train_graph_dir, exist_ok=True)
os.chdir('../rl-baselines-zoo')
print("Training model")
subprocess.check_output(['python', 'train.py', '--algo', args.algo, '--env', args.env, '-q', str(args.quant_bit_width), '--quant-delay', str(args.quant_delay), '-n', str(args.n), '-f', args.base])

os.chdir('../quant-scripts')
eval_graph_dir = os.path.join(args.base, "quant_train/eval", args.algo)
os.makedirs(eval_graph_dir, exist_ok=True)
print("Creating Quantized Eval Graph")
subprocess.check_output(['python','create_eval_graph.py','--algo', args.algo, '--env', args.env, '-q', str(args.quant_bit_width),'--base', args.base])

frozen_graph_def = os.path.join(args.base, 'frozen_qt', args.algo, args.env + ".pb")
os.makedirs("/".join(frozen_graph_def.split("/")[:-1]), exist_ok=True)
print("Freezing Graph")
subprocess.check_output(['./freeze_qt.sh', args.algo, args.env, output_node, args.base])
print("Creating TFLite File")
if is_atari:
    subprocess.check_output(['tflite_convert','--output_file=/tmp/foo.tflite', '--graph_def_file='+frozen_graph_def, '--input_arrays='+input_node, '--output_arrays='+output_node, '--target_ops=TFLITE_BUILTINS', '--inference_type=FLOAT','--inference_input_type=QUANTIZED_UINT8', '--mean_value=128', '--std_dev_values=127'])
else:
    subprocess.check_output(['tflite_convert','--output_file=/tmp/foo.tflite', '--graph_def_file='+frozen_graph_def, '--input_arrays='+input_node, '--output_arrays='+output_node, '--target_ops=TFLITE_BUILTINS', '--inference_type=FLOAT'])

print("Evaluating model")
tflite_model_path = "/tmp/foo.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

model_path = os.path.join(train_graph_dir, args.env + ".zip")
print("Loading model from", model_path)
log_dir = "/tmp"
hyperparams, stats_path = get_saved_hyperparams(model_path, norm_reward=False, test_mode=True)

print("Running ", args.algo, " on ", args.env)
set_global_seeds(0)
env = create_test_env(args.env, n_envs=1, is_atari=is_atari,
                          stats_path=stats_path, seed=0, log_dir=log_dir,
                          should_render=False,
                          hyperparams=hyperparams)

print("Evaluating converted model")
episode_rewards, lengths, norm_rewards = [], [], []
for i in range(100):
    obs, done = env.reset(), False
    episode_rew, norm_rew, length = 0, 0.0, 0
    while not done:
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

directory = os.path.join(quarl_directory, "csvs/qt", str(args.quant_bit_width), args.algo, args.env)
os.makedirs(directory, exist_ok=True)
with open(directory + "rewards.csv", 'w') as file:
    writer = csv.writer(file, delimiter=',',  quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(episode_rewards)


