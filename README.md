# Quantized Reinforcement Learning (QuaRL)

Code for QuaRL, a framework for evaluating the effects of quantization on reinforcement learning policies across different environments, training algorithms and quantization methods. 

![](https://github.com/harvard-edge/quarl/blob/master/docs/QuaRL-intro-figure.png)

| Environments       | Reinforcement Learning Algorithms           | Quantization Methods  |
| ------------- |:-------------:| -----:|
| Atari Games      | Proximal Policy Optimization (PPO) | Post-training Quantization |
| OpenAI Gym     | Actor Critic (A2C)     |   Quantization Aware Training |
| PyBullet | Deep Deterministic Policy Gradients (DDPG)     |     |
|           | DQN (Deep Q Networks)           | |

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

For example, here is an example of visualizing the weights distribution for breakout envionment trained using DQN, PPO, and A2C:
![](https://github.com/harvard-edge/quarl/blob/master/docs/breakout-weight-distribution.png)

## Results
For results, please check our [paper](https://arxiv.org/abs/1910.01055) 
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
