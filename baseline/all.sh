#!/bin/bash

bash benchmark.sh dqn SeaquestNoFrameskip-v4
bash benchmark.sh dqn PongNoFrameskip-v4
bash benchmark.sh dqn SpaceInvadersNoFrameskip-v4
bash benchmark.sh dqn QbertFrameskip-v4
bash benchmark.sh dqn BeamRiderNoFrameskip-v4
bash benchmark.sh dqn MsPacmanNoFrameskip-v4
bash benchmark.sh dqn EnduroNoFrameskip-v4
bash benchmark.sh dqn LunarLander-v2
bash benchmark.sh dqn Acrobot-v1
bash benchmark.sh dqn MountainCar-v0

bash benchmark.sh ppo2 BreakoutNoFrameskip-v4
bash benchmark.sh ppo2 SeaquestNoFrameskip-v4
bash benchmark.sh ppo2 PongNoFrameskip-v4
bash benchmark.sh ppo2 SpaceInvadersNoFrameskip-v4
bash benchmark.sh ppo2 QbertFrameskip-v4
bash benchmark.sh ppo2 BeamRiderNoFrameskip-v4
bash benchmark.sh ppo2 MsPacmanNoFrameskip-v4
bash benchmark.sh ppo2 EnduroNoFrameskip-v4
bash benchmark.sh ppo2 LunarLander-v2
bash benchmark.sh ppo2 Acrobot-v1
bash benchmark.sh ppo2 MountainCar-v0
bash benchmark.sh ppo2 MountainCarContinuous-v0
bash benchmark.sh ppo2 BipedalWalker-v2

bash benchmark.sh a2c BreakoutNoFrameskip-v4
bash benchmark.sh a2c SeaquestNoFrameskip-v4
bash benchmark.sh a2c PongNoFrameskip-v4
bash benchmark.sh a2c SpaceInvadersNoFrameskip-v4
bash benchmark.sh a2c QbertFrameskip-v4
bash benchmark.sh a2c BeamRiderNoFrameskip-v4
bash benchmark.sh a2c MsPacmanNoFrameskip-v4
bash benchmark.sh a2c EnduroNoFrameskip-v4
bash benchmark.sh a2c LunarLander-v2
bash benchmark.sh a2c Acrobot-v1
bash benchmark.sh a2c MountainCar-v0
bash benchmark.sh a2c MountainCarContinuous-v0
bash benchmark.sh a2c BipedalWalker-v2

bash benchmark.sh ddpg MountainCarContinuous-v0
bash benchmark.sh ddpg BipedalWalker-v2
bash benchmark.sh ddpg LunarLanderContinuous-v2
bash benchmark.sh ddpg HalfCheetahBulletEnv-v0
bash benchmark.sh ddpg Walker2DBulletEnv-v0
bash benchmark.sh ddpg AntBulletEnv-v0
