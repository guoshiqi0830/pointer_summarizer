import os

from torch._C import ThroughputBenchmark

root_dir = os.path.expanduser("/content/drive/MyDrive/Experiment/20210816/")
data_dir = os.path.expanduser("/content/drive/MyDrive/Experiment/20201213/")

train_data_path = os.path.join(data_dir, "data/finished_files/chunked/train_*")
eval_data_path = os.path.join(data_dir, "data/finished_files/val.bin")
decode_data_path = os.path.join(data_dir, "data/finished_files/test.bin")
vocab_path = os.path.join(data_dir, "data/finished_files/vocab")
log_root = os.path.join(root_dir, "log")

# Hyperparameters
hidden_dim= 256
emb_dim= 128
batch_size= 8
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

use_gpu=True

lr_coverage=0.15

DEBUG=False
BREAK_POINT=1
REC_ENTROPY=False

# Loss Mask
loss_mask=False
alpha=4

# Sparse Loss
tsallis_alpha=1
temperature=1

# adaptive sparsemax
adaptive_sparsemax = True
fix_the_rest = False

# top_p
use_top_p = False
top_p = 0.99
