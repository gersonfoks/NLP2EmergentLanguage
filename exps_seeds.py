from att_game_pl_config import run

import torch
import pytorch_lightning as pl

import argparse
import yaml
parser = argparse.ArgumentParser(description='Run an experiment defined an a yml file')

parser.add_argument('--config', default="config/example_config.yml", required=False)

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f)

for i in range(5):
    pl.seed_everything(i)
    run(config["n_attributes"], config["attributes_size"], config["n_receiver"], 
    config["n_symbols"], config["msg_len"], 
        config["samples_per_epoch_train"], 
        config["samples_per_epoch_test"], 
        config["max_epochs"], config["fixed_size"], 
        config["pretrain_n_epochs"], config["learning_rate"])