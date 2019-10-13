import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import getopt
from quantize_uniform import quantize_uniform

class Visualize_Pkl():
	def __init__(self, path, num_bits):
		self.path = path
		self.num_bits = num_bits

	def output(self):
		files = os.listdir(self.path)
		models = []
		for i in files:
			if '.pkl' in i:
				models.append(i)
		os.chdir(self.path)
		for file in models:
			if '.pkl' in file:
				self.visualize(file)

	def load_weights_from_pkl(self, load_file):
	    if isinstance(load_file, str):
	    	if not os.path.exists(load_file):
	    		if os.path.exists(load_file + ".pkl"):
	    			load_file += ".pkl"
	    		else:
	    			raise ValueError("Error: the file {} could not be found".format(load_file))
	    	with open(load_file, "rb") as file:
	    		data, params = pickle.load(file)
	    else:
	    	data, params = pickle.load(load_file)

	    flattened = [x.flatten().tolist() for x in params]
	    a = []
	    for x in flattened:
	        a += x
	    return a

	def visualize(self, file):
		a = self.load_weights_from_pkl(file)
		file = file[:-4]
		x = np.asarray(a, dtype=np.float32)
		q = quantize_uniform(x, num_bits=self.num_bits)
		q_uni = np.unique(q)
		f = open("output.txt", "a")
		print(file,":","weight_bias_min:", x.min(),", weight_bias_max:", x.max(), ", range:", x.max()-x.min(), file = f)
		f.close()
		print(file,":","weight_bias_min:", x.min(),", weight_bias_max:", x.max(), ", range:", x.max()-x.min())
		plt.figure(figsize=(8,4))
		plt.hist(x, range=(x.min(), x.max()),bins=100, density=0, facecolor="blue", edgecolor="black", log=True)
		for xc in q_uni:
			plt.axvline(x=xc, color="orange", alpha=0.4)
		plt.xlabel("Weight and Bias")
		plt.ylabel("Frequency")
		plt.title(file + " weight & bias distribution")
		plt.savefig(file + ".pdf", bbox_inches='tight')
		# plt.show()
	

def main(argv):
	path = 'example_folder_pkl/'
	num_bits = 8
	try:
		opts, args = getopt.getopt(argv, "hf:b:", ["help", "folder=", "num_bits="])
	except getopt.GetoptError:
		print("Error: visualize_pkl.py -f <folder> -b <num_bits>")
		print("   or: visualize_pkl.py --folder=<folder> --num_bits=<num_bits>")
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print("Error: visualize_pkl.py -f <folder> -b <num_bits> ")
			print("   or: visualize_pkl.py --folder=<folder> --num_bits=<num_bits>")
			sys.exit()
		elif opt in ("-f", "--folder"):
			path = arg
		elif opt in ("-b", "--num_bits"):
			num_bits = int(arg)
	print("folder: ",path)
	s = Visualize_Pkl(path = path, num_bits=num_bits)
	s.output()
	 
if __name__ == '__main__':
    main(sys.argv[1:])
