#!/bin/sh

base="$5/saved/$4/$1/$2"
freeze_graph \
--input_graph="$base.pb" \
--input_checkpoint="$base.ckpt" \
--output_graph="$5/frozen/$4/$1/$2.pb" --output_node_names="$3"

