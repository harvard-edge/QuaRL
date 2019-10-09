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
