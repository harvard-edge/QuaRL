import tensorflow as tf
from tensorflow.python.framework import tensor_util
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import getopt
from quantize_uniform import quantize_uniform


class Visualize_Pb():
    def __init__(self, path, num_bits):
        self.path = path
        self.num_bits = num_bits

    def output(self):
        files = os.listdir(self.path)
        models = []
        for i in files:
            if '.pb' in i:
                models.append(i)
        os.chdir(self.path)
        for file in models:
            if '.pb' in file:
                self.visualize(file)

    def load_model(self, load_file):
        with tf.gfile.GFile(load_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
            graph_nodes=[n for n in graph_def.node]
        return graph, graph_nodes

    def visualize(self, file):
        graph, graph_nodes = self.load_model(file)
        wts = [n for n in graph_nodes if n.op=='Const']
        weight_bias = []
        file = file[:-3]
        for n in wts:
            if '/min' not in n.name and '/max' not in n.name:
                if '/w' in n.name or '/weights' in n.name or '/b' in n.name or '/bias' in n.name:
                    param = tensor_util.MakeNdarray(n.attr['value'].tensor)
                    if '/c' in n.name:
                        for k in param:
                            for l in k:
                                for p in l:
                                    for u in p:
                                        weight_bias.append(u)
                    elif '/w' in n.name or '/weights' in n.name:
                        for k in param:
                            for l in k:
                                weight_bias.append(l)
                    elif '/b' in n.name or '/bias' in n.name:
                        for k in param:
                            weight_bias.append(k)
        x = np.asarray(weight_bias, dtype=np.float32)
        q = quantize_uniform(x, num_bits = self.num_bits)
        q_uni = np.unique(q)
        f = open("output.txt", "a")
        print(file, ":", "weight_bias_min:", x.min(), ", weight_bias_max:", x.max(), ", range:", x.max()-x.min(), file = f)
        f.close()
        print(file, ":", "weight_bias_min:", x.min(), ", weight_bias_max:", x.max(), ", range:", x.max()-x.min())
        plt.figure(figsize=(8, 4))
        plt.hist(x, range=(x.min(), x.max()), bins=100, density=0, facecolor="blue", edgecolor="black", log=True)
        for xc in q_uni:
            plt.axvline(x = xc, color = "orange", alpha = 0.4)
        plt.xlabel("Weight and Bias")
        plt.ylabel("Frequency")
        plt.title(file + ' weight & bias distribution')
        plt.savefig(file + ".pdf", bbox_inches='tight')
        # plt.show()

def main(argv):
    path = 'example_folder_pb/'
    num_bits = 8
    try:
        opts, args = getopt.getopt(argv, "hf:b:", ["help", "folder=", "num_bits="])
    except getopt.GetoptError:
        print("Error: visualize_pb.py -f <folder> -b <num_bits>")
        print("   or: visualize_pb.py --folder=<folder> --num_bits=<num_bits>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Error: visualize_pb.py -f <folder> -b <num_bits>")
            print("   or: visualize_pb.py --folder=<folder> --num_bits=<num_bits>")
            sys.exit()
        elif opt in ("-f", "--folder"):
            path = arg
        elif opt in ("-b", "--num_bits"):
            num_bits = int(arg)
    print("folder: ", path)
    s = Visualize_Pb(path=path, num_bits=num_bits)
    s.output()

if __name__ == '__main__':
    main(sys.argv[1:])

