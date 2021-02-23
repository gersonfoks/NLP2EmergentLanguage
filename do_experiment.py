import torch
from numpy import infty
import numpy as np

import pytorch_lightning as pl

import argparse
import yaml

from experiment_utils import run_game_with_config, create_name, result_to_file, get_summary_results

parser = argparse.ArgumentParser(description='Run a grid defined in a given ')

parser.add_argument('--config', default="config/example_experiment.yaml", required=False)

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f)





results = { metric: [] for metric in config["metrics"]}


for i in range(config["n_runs"]):
    print(config)
    pl.seed_everything(i)
    result_run = run_game_with_config(config)
    print(result_run)
    for metric in config["metrics"]:
        if isinstance(result_run[metric], torch.Tensor):

            results[metric].append(result_run[metric].item())
        else:
            results[metric].append(result_run[metric])


name = create_name(config)

result_to_file(name, results)
results_summary = get_summary_results(name)

print(results_summary)


print(results)







