#!/bin/sh
../../tf/tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph="/home/vj-reddi/quantization-air-learning/frozen_qt/$1/$2.pb" \
--out_graph="/home/vj-reddi/quantization-air-learning/quant/$1/$2.pb" \
--inputs="$3" \
--outputs="$4" \
--transforms="quantize_weights"
