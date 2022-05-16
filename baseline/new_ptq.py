import tensorflow as tf
import pkg_resources
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
import math
from math import log2
from stable_baselines import bench, logger
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines.common import tf_util, set_global_seeds
import scipy.stats as ss
import scipy

if pkg_resources.get_distribution("stable_baselines").version >= "2.6.0":
    sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.deepq.replay_buffer
    stable_baselines.deepq.replay_buffer.Memory = stable_baselines.deepq.replay_buffer.ReplayBuffer

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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
        kl = scipy.stats.entropy(pg(p),qg(q))
        print("p,q",scipy.stats.entropy(pg(p),qg(q)))
        print("len of p",len(p))
        return kl
    else:
        return 0

def Q(W, n):
    w_old = W
    if n >= 32:
        return W
    assert(len(W.shape) <= 2)
    range = np.abs(np.min(W)) + np.abs(np.max(W))
    d = range / (2**(n))
    z = -np.min(W, 0) // d
    W = np.rint(W / d)
    W = d * (W)
    
    kl = kl_scipy(w_old, W)
    
    # plot histogram
    import matplotlib.pyplot as plt
    plt.hist(w_old.flatten(), bins=100)
    plt.hist(W.flatten(), bins=100)
    
    # save plot
    plt.savefig('histogram.png')
    
    return W, kl
    
def check_if_convolution(var, convolutions):
    for i in convolutions:
        if i in var:
            return True
    return False

def conv_Q(W, n):
    if n >= 32:
        return W
    w_old = W
    newweight = np.zeros_like(W)
    # print(W.shape)
    for i in range(W.shape[-1]):
        range_i = np.abs(np.min(W[:,:,:,i])) + np.abs(np.max(W[:,:,:,i]))
        d = range_i / (2**(n))
        z = -np.min(W[:,:,:,i], 0) // d
        temp = np.rint(W[:,:,:,i] / d)
        newweight[:,:,:,i] += d * temp
    kl= kl_scipy(w_old, newweight)
    return newweight, kl

if __name__=="__main__":

    kl_array = []
    
    sys.path.insert(0, './rl-baselines-zoo/')
    file_dir = os.path.dirname(os.path.abspath(__file__))
    quarl_directory = "/".join(file_dir.split("/")[:-1])
    from utils import create_test_env, get_saved_hyperparams, ALGOS

    algo = sys.argv[1]
    env = sys.argv[2]
    class Algo(ALGOS[algo]):
        def __init__(self, policy, env, **kwargs):
            super(Algo, self).__init__(policy=policy, env=env, **kwargs)

        def save_graph(self, dir, name):
            with self.graph.as_default():
                saver = tf.train.Saver()
                tf.train.write_graph(self.sess.graph_def, dir, name + '.pb')
                saver.save(self.sess, dir + '/' + name + '.ckpt')
    print("quarl_directory: ", quarl_directory)
    models_path = os.path.join(quarl_directory, "baseline/original_agents/trained_agents")
    model_path = os.path.join(models_path, algo, env + ".pkl")
    og_model = Algo.load(model_path)

    convolutions = set(['/c1','/c2','/c3'])
    weights, qweights, neww = {}, {}, {}
    with og_model.sess as session:
        vars = tf.trainable_variables()
        # print(vars)
        assign_ops = []
        for var in vars:
            if algo != 'dqn':
                if 'pi' in var.name:
                    if ('w' in var.name or 'weight' in var.name): 
                        print("Quantizing {}".format(var.name))
                        weights[var.name] = session.run(var.value())

                        if sys.argv[3] == '16':
                            qweights[var.name] = weights[var.name].astype(np.float16).astype(np.float32)
                        else:
                            if check_if_convolution(var.name, convolutions):
                                qweights[var.name], kl = conv_Q(weights[var.name], int(sys.argv[3]))
                                kl_array.append(kl)
                            else:
                                
                                qweights[var.name],kl = Q(weights[var.name], int(sys.argv[3]))
                                kl_array.append(kl)
                        assign_ops.append(var.assign(qweights[var.name]))
                # DDPG policy does not have "w" in the var.name
                # This hack is for DDPG
                else:
                    weights[var.name] = session.run(var.value())
                    if sys.argv[3] == '16':
                        qweights[var.name] = weights[var.name].astype(np.float16).astype(np.float32)
                    else:
                            if check_if_convolution(var.name, convolutions):
                                qweights[var.name], kl = conv_Q(weights[var.name], int(sys.argv[3]))
                                kl_array.append(kl)
                            else:
                                
                                qweights[var.name],kl = Q(weights[var.name], int(sys.argv[3]))
                                kl_array.append(kl)
                    assign_ops.append(var.assign(qweights[var.name]))
            else:
                if 'state_value' in var.name or 'action_value' in var.name:
                    if ('w' in var.name or 'weight' in var.name): 
                        print("Quantizing {}".format(var.name))
                        weights[var.name] = session.run(var.value())
                        if sys.argv[3] == '16':
                            qweights[var.name] = weights[var.name].astype(np.float16).astype(np.float32)
                        else:
                            if check_if_convolution(var.name, convolutions):
                                qweights[var.name], kl = conv_Q(weights[var.name], int(sys.argv[3]))
                                kl_array.append(kl)
                            else:
                                qweights[var.name], kl = Q(weights[var.name], int(sys.argv[3]))
                                kl_array.append(kl)
                        assign_ops.append(var.assign(qweights[var.name]))

        for assign_op in assign_ops:
            session.run(assign_op)

        save_path = 'quantized/{}/{}'.format(sys.argv[3], algo)
        os.makedirs(save_path, exist_ok=True)

        os.system("rm -r {}/{}".format(save_path, env))
        os.system("cp -r {} {}/".format(model_path[:-4], save_path))
        og_model.save('{}/{}.pkl'.format(save_path, env))
        print("Kl divergence: ", sum(kl_array))
        
        

