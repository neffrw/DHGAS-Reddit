import subprocess
import os

models = [
    {"model": "LSTM", "device": 1},
    {"model": "GCN", "device": 1},
    {"model": "GAT", "device": 1},
    {"model": "RGCN", "device": 1},
    {"model": "HGT", "device": 1},
    {"model": "HTGNN", "device": "cpu"},
]

# models = ["GCN", "GAT", "RGCN", "HGT", "HTGNN"]
dims = [8, 16, 32]
seeds = [22, 48304, 1937475]

dataset = "Reddit-troll"
patience = 100
log_dir = "logs"

for model_config in models:
    for dim in dims:
        for seed in seeds:
            model = model_config["model"]
            hid_dim = dim
            out_dim = dim
            device = model_config["device"]
            if model == "HTGNN":
                n_heads = 1
            else:
                n_heads = 4
            
            command = f"python scripts/run/run_model.py --dataset {dataset} --model {model} --device {device} --patience {patience} --hid_dim {hid_dim} --out_dim {out_dim} --n_heads {n_heads} --log_dir {log_dir}/{dataset}/{model}/{hid_dim}/{seed} --seed {seed}"
            info_file = f"{log_dir}/{dataset}/{model}/{hid_dim}/{seed}/info.json"

            if not os.path.exists(info_file):
                print(" Running ", command)
                subprocess.run(command, shell=True)
            else:
                print("Skip", command)
