import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--n_actors", default=1, type=int)
parser.add_argument("--taskstr", default="gym,MountainCarContinuous-v0")
parser.add_argument("--logpath", default="logfiles_debug")
parser.add_argument("--port", default=8181, type=int)
parser.add_argument("--broadcaster_table_name", default="broadcaster_table")
parser.add_argument("--replay_table_name", default="priority_table")
parser.add_argument("--model_table_name", default="model_table_name")
parser.add_argument("--n_step", default=5, type=int)
parser.add_argument("--discount", default=.99, type=float)
parser.add_argument("--sigma", default=.01, type=float)
parser.add_argument("--num_episodes", default=10000, type=int)
parser.add_argument("--replay_table_max_times_sampled", default=8, type=int)
parser.add_argument("--replay_table_max_replay_size", default=200000, type=int)
#parser.add_argument("--replay_table_max_replay_size", default=1000, type=int)
parser.add_argument("--shutdown_table_name", default="shutdown_table_name")
parser.add_argument("--learner_device_placement", default="GPU")
parser.add_argument("--actor_device_placement", default="CPU")
parser.add_argument("--actor_id", default=0, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--inference_mode", choices=["tensorflow", "pytorch"], default="pytorch")
parser.add_argument("--quantize", type=int, choices=[8,16,32], default=32)
parser.add_argument("--quantize_communication", type=int, default=32)
parser.add_argument("--model_str", type=str, default="4096,4096,4096")
parser.add_argument("--actor_update_period", type=int, default=100)
parser.add_argument("--min_replay_size", type=int, default=100000)
parser.add_argument("--compress", type=int, default=0)

# Whether to compress weights of the state dict (vs compress the pickle dumped object).
# If use, use with quantize_communication = 0, as this compression method is lossy. Use with 8.
parser.add_argument("--weight_compress", type=int, default=0)
