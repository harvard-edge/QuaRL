#!/bin/sh
path="$4/quant_train/eval/$1/$2"
freeze_graph \
--input_graph="$path.pb" \
--input_checkpoint="$path.ckpt" \
--output_graph="$4/frozen_qt/$1/$2.pb" --output_node_names="$3"
