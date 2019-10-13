### Visualizing
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
![](https://github.com/harvard-edge/quarl/blob/master/docs/breakout-weight-distribution.png)
