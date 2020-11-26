#!/bin/bash

python collate.py dqn SeaquestNoFrameskip-v4
python collate.py dqn PongNoFrameskip-v4
python collate.py dqn SpaceInvadersNoFrameskip-v4
python collate.py dqn QbertFrameskip-v4
python collate.py dqn BeamRiderNoFrameskip-v4
python collate.py dqn MsPacmanNoFrameskip-v4
python collate.py dqn EnduroNoFrameskip-v4
python collate.py dqn LunarLander-v2
python collate.py dqn Acrobot-v1
python collate.py dqn MountainCar-v0

python collate.py ppo2 BreakoutNoFrameskip-v4
python collate.py ppo2 SeaquestNoFrameskip-v4
python collate.py ppo2 PongNoFrameskip-v4
python collate.py ppo2 SpaceInvadersNoFrameskip-v4
python collate.py ppo2 QbertFrameskip-v4
python collate.py ppo2 BeamRiderNoFrameskip-v4
python collate.py ppo2 MsPacmanNoFrameskip-v4
python collate.py ppo2 EnduroNoFrameskip-v4
python collate.py ppo2 LunarLander-v2
python collate.py ppo2 Acrobot-v1
python collate.py ppo2 MountainCar-v0
python collate.py ppo2 MountainCarContinuous-v0
python collate.py ppo2 BipedalWalker-v2

python collate.py a2c BreakoutNoFrameskip-v4
python collate.py a2c SeaquestNoFrameskip-v4
python collate.py a2c PongNoFrameskip-v4
python collate.py a2c SpaceInvadersNoFrameskip-v4
python collate.py a2c QbertFrameskip-v4
python collate.py a2c BeamRiderNoFrameskip-v4
python collate.py a2c MsPacmanNoFrameskip-v4
python collate.py a2c EnduroNoFrameskip-v4
python collate.py a2c LunarLander-v2
python collate.py a2c Acrobot-v1
python collate.py a2c MountainCar-v0
python collate.py a2c MountainCarContinuous-v0
python collate.py a2c BipedalWalker-v2

python collate.py ddpg MountainCarContinuous-v0
python collate.py ddpg BipedalWalker-v2
python collate.py ddpg LunarLanderContinuous-v2
python collate.py ddpg HalfCheetahBulletEnv-v0
python collate.py ddpg Walker2DBulletEnv-v0
python collate.py ddpg AntBulletEnv-v0

