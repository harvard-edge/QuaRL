# Quantized Reinforcement Learning (QuaRL)
**Stable Code Release**:9th October 2019

Code for QuaRL, a framework for evaluating the effects of quantization on reinforcement learning policies across different environments, training algorithms and quantization methods. 

**Supported Environments**
* Atari
* OpenAI Gym 
* PyBullet

**Supported Reinforcement Learning Algorithms**
* Proximal Policy Optimization (PPO)
* Actor Critic (A2C)
* Deep Deterministic Policy Gradients (DDPG)
* DQN (Deep Q Networks)

**Supported quantization methods**
* Post-training Quantization
* Quantization Aware Training

Read the paper here for more information: https://arxiv.org/abs/1910.01055

# Table of Contents
1. [Introduction](#Introduction)
2. [Quickstart](#Quickstart)
3. [Results](#Results)
4. [Citations](#Citations)

## Introduction
Deep reinforcement learning is used for many tasks including game playing, robotics and transportation. However, deep reinforcement learning policies are extremely resource intensive due to the computationally expensive nature of the neural networks that power them. The computationally expensive nature of these policies not only make training slow and expensive, but also hinder deployment on resource limited devices like drones.

One solution to improving neural network performance is quantization, a method that reduces the precision of neural network weights to enable training and inference with fast low-precision operations. Motivated by recent trends demonstrating that image models may be quantized to < 8 bits without sacrificing performance, we investigate whether the same is true for reinforcement learning models.

To that end, we introduce a framework for training, quantizing and evaluating the effects of different quantization methods on various reinforcement learning tasks and training algorithms. This code forms the backbone of the experimental setup used for our paper (https://arxiv.org/abs/1910.01055). 

## Quickstart

### Training

Full Precision Training:

```
python train.py --method DQN --task breakout --precision 32 --quantization_method quantization_aware --output_path dqn_breakout_quantaware_precision=32
```

8-bit Quantization Aware Training:

```
python train.py --method DQN --task breakout --precision 8 --quantization_method quantization_aware --output_path dqn_breakout_quantaware_precision=8
```

8-bit Post-training Quantization:

```
python train.py --method DQN --task breakout --precision 8 --quantization_method post_train_quantization --output_path dqn_breakout_posttrainquant_precision=8
```

### Evaluating

```
python evaluate.py --task breakout --input_path dqn_breakout_posttrainquant_precision=8
```

### Help
```
python train.py --help
```

### Visualizing
Visualizing the model's parameter (weight & bias) distribution.

If the saved model is in '.pb' format, please run 
```
python visualize_pb.py -f <folder>
or: python visualize_pb.py --folder=<folder>
```

If the saved model is in '.pkl' format, please run 
```
python visualize_pkl.py -f <folder>
or: python visualize_pkl.py --folder=<folder>
```

The parameter distribution plot will be saved under ```<folder>```, and the detailed statistical information will be saved in ```output.txt``` under ```<folder>```.

## Results
## Citations
