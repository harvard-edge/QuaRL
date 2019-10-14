<p align="center">
  <img src="https://github.com/harvard-edge/quarl/blob/master/docs/QuaRL.jpg">
</p>

# Quantized Reinforcement Learning (QuaRL)

Code for QuaRL, a framework for evaluating the effects of quantization on reinforcement learning policies across different environments, training algorithms and quantization methods. 

# Table of Contents
1. [Introduction](#Introduction)
2. [Quickstart](#Quickstart)
3. [Results](#Results)
4. [Citations](#Citations)

## Introduction

Deep reinforcement learning is used for many tasks including game playing, robotics and transportation. However, deep reinforcement learning policies are extremely resource intensive due to the computationally expensive nature of the neural networks that power them. The computationally expensive nature of these policies not only make training slow and expensive, but also hinder deployment on resource limited devices like drones.

One solution to improving neural network performance is quantization, a method that reduces the precision of neural network weights to enable training and inference with fast low-precision operations. Motivated by recent trends demonstrating that image models may be quantized to < 8 bits without sacrificing performance, we investigate whether the same is true for reinforcement learning models.

To that end, we introduce the end-to-end framework (shown below) for training, quantizing and evaluating the effects of different quantization methods on various reinforcement learning tasks and training algorithms. This code forms the backbone of the experimental setup used for our paper (https://arxiv.org/abs/1910.01055). 

![](https://github.com/harvard-edge/quarl/blob/master/docs/QuaRL-intro-figure.png)

The framework currently support the following environments, RL algorithms and quantization methods.

#### Environments
- Atari Games
- OpenAI Gym
- PyBullet
#### RL Algorithms
- Proximal Policy Optimization (PPO)
- Actor Critic (A2C)
- Deep Deterministic Policy Gradients (DDPG)
- DQN (Deep Q Networks)
#### Quantization Methods
- Post-training Quantization
- Quantization Aware Training

Read the paper here for more information: https://arxiv.org/abs/1910.01055

## Quickstart
We suggest that you create an environment using conda first
```
conda create --name quarl python=3.6
conda activate quarl
```
For ubuntu:
```
./setup_ubuntu.sh
cd quant-scripts
```
For MacOS:
```
./setup_mac.sh
cd quant-scripts
```

### 8-bit Post-training Quantization:

```
python ptq.py --algo dqn --env BreakoutNoFrameskip-v4 --int 1
```
### fp16 Post-training Quantization:

```
python ptq.py --algo dqn --env BreakoutNoFrameskip-v4 --fp16 1
```
### Run fp32 model using TFLite (as a control experiment):

```
python ptq.py --algo dqn --env BreakoutNoFrameskip-v4 --fp32 1
```
### 8-bit Quantization Aware Training and testing:

QAT usually requires training a model from scratch. We suggest setting quant-delay as half the total number of training steps. The official TF guidelines suggest finetuning min, max quantization ranges after the model has fully converged but in the case of RL over-training usually results in bad performance. QAT results also vary a lot depending on training so exact rewards as mentioned in the paper are not always guaranteed.

```
python qat.py --algo dqn --env BreakoutNoFrameskip-v4 -q 7 --quant-delay 5000000 -n 10000000
```

### Visualization
Visualizing the model's parameter (weight & bias) distribution.

If the saved model is in '.pb' format, please run 
```
python visualize_pb.py -f <folder> -b <num_bits>
or: python visualize_pb.py --folder=<folder> --num_bits=<num_bits>
```

If the saved model is in '.pkl' format, please run 
```
python visualize_pkl.py -f <folder> -b <num_bits>
or: python visualize_pkl.py --folder=<folder> --num_bits=<num_bits>
```

The parameter distribution plot will be saved under ```<folder>```, and the detailed statistical information will be saved in ```output.txt``` under ```<folder>```.

For example, here is an example of visualizing the weights distribution for breakout envionment trained using DQN, PPO, and A2C:
<p align="center">
  <img src="https://github.com/harvard-edge/quarl/blob/master/docs/breakout-weight-distribution.png" width=400>
</p>

## Results
For results, please check our [paper](https://arxiv.org/abs/1910.01055). 

## Citations
To cite this repository in publications:
```
@misc{quantized-rl,
    title={Quantized Reinforcement Learning (QUARL)},
    author={Srivatsan Krishnan and Sharad Chitlangia and Maximilian Lam and Zishen Wan and Aleksandra Faust and Vijay Janapa Reddi},
    year={2019},
    eprint={1910.01055},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
