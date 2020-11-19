import os
import sys
import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile
from stable_baselines.common import tf_util
from google.protobuf import text_format
sys.path.insert(0, '../rl-baselines-zoo/')
from utils import ALGOS, get_latest_run_id

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, nargs='+', default="CartPole-v1", help='environment ID(s)')
parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                        type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('-q', '--quant-bit-width', help='Set bit width for quantization', default=8, type=int)
parser.add_argument('--base', help="Set base directory for saved models", default="/".join(os.path.dirname(os.path.abspath(__file__))[:-1]), type=str)
args = parser.parse_args()

quant_train = "eval"
log_path = os.path.join(args.base, "quant_train/train/", str(args.quant_bit_width),args.algo)
path = os.path.join(log_path, "{}_{}".format(args.env[0], get_latest_run_id(log_path, args.env[0])), args.env[0] + ".pb")

print("Loading models from ", path, "\n")
class Algo(ALGOS[args.algo]):
    def __init__(self, policy, env, **kwargs):
        super(Algo, self).__init__(policy=policy, env=env, **kwargs)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        data, params = cls._load_from_file(load_path)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=None, _init_setup_model=False, w_bits=args.quant_bit_width, act_bits=args.quant_bit_width, quant_train="eval")
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model.set_env(env)
        model.setup_model()
        v1 = tf.get_variable("v1", shape=[3])
        #tf.contrib.quantize.create_eval_graph(input_graph = model.graph)

        model.saver = tf.train.Saver()

        model.saver.restore(model.sess, load_path[:-3] + "ckpt")
        model.load_parameters(params)
        model.file_writer = tf.summary.FileWriter("/tmp/pb", model.sess.graph)
        return model


model = Algo.load(path[:-2] + "zip")
eval_graph_file = os.path.join(args.base, "quant_train/eval/", args.algo, args.env[0] + ".pb")

one, second = "/".join(eval_graph_file.split("/")[:-1]), eval_graph_file.split("/")[-1]
print("Saving to", eval_graph_file[:-3])
with model.sess.as_default():
    tf.train.write_graph(model.sess.graph_def, one, second)
    # with self.sess.graph.as_default()
    #saver = tf.train.Saver()
    model.saver.save(model.sess, eval_graph_file[:-2] + "ckpt")
