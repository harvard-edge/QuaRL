#!/bin/bash

set -e

for i in 32 30 28 26 24 22 20 18 16 14 12 10 8 6 4 2; do
~/anaconda3/envs/quarl/bin/python new_ptq.py $1 $2 $i;
    for j in 0 2 4; do
        ~/anaconda3/envs/quarl/bin/python rl-baselines-zoo/enjoy.py --algo $1 --env $2 --no-render --folder ./quantized/$i/ -n 15000 --reward-log csvs/$i/$1/$2/$j/ --deterministic --seed $j;
    done
done