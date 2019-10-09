import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import getopt

class Visualize_Pkl():
	def __init__(self, path):
		self.path = path

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
		f = open("output.txt", "a")
		print(file,":","weight_bias_min:", x.min(),", weight_bias_max:", x.max(), ", range:", x.max()-x.min(), file = f)
		f.close()
		print(file,":","weight_bias_min:", x.min(),", weight_bias_max:", x.max(), ", range:", x.max()-x.min())
		plt.figure(figsize=(8,4))
		plt.hist(x, range=(x.min(), x.max()),bins=100, density=0, facecolor="blue", edgecolor="black", log=True)
		plt.xlabel("Weight and Bias")
		plt.ylabel("Frequency")
		plt.title(file + " weight & bias distribution")
		plt.savefig(file + ".pdf", bbox_inches='tight')
		# plt.show()
	

def main(argv):
	path = 'example_folder_pkl/'
	try:
		opts, args = getopt.getopt(argv, "hf:", ["help", "folder="])
	except getopt.GetoptError:
		print("Error: visualize_pkl.py -f <folder>")
		print("   or: visualize_pkl.py --folder=<folder>")
		sys.exit(2)
	for opt, arg in opts:
		if opt in ("-h", "--help"):
			print("Error: visualize_pkl.py -f <folder>")
			print("   or: visualize_pkl.py --folder=<folder>")
			sys.exit()
		elif opt in ("-f", "--folder"):
			path = arg
	print("folder: ",path)
	s = Visualize_Pkl(path = path)
	s.output()
	 
if __name__ == '__main__':
    main(sys.argv[1:])
