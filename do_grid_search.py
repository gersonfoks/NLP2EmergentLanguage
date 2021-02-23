from itertools import product

import torch
import pytorch_lightning as pl
from numpy import infty
import numpy as np
from attribute_game.pl_model import AttributeBaseLineModel, AttributeModelWithPrediction
from attribute_game.utils import get_sender, get_receiver, get_predictor
from callbacks.msg_callback import MsgCallback, MsgFrequencyCallback, EntropyMeasure, MeasureCallbacks, \
    ResetDatasetCallback, DistinctSymbolMeasure

from att_game_pl_config import run

import torch
import pytorch_lightning as pl

import argparse
import yaml

from experiment_utils import construct_configs, run_game_with_config, print_best_pretty
from utils import cross_entropy_loss_2

parser = argparse.ArgumentParser(description='Run a grid defined in a given ')

parser.add_argument('--config', default="config/gridsearch_config_example.yaml", required=False)

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f)



configs = construct_configs(config)

lowest_loss = infty
best = None

for config in configs:
    loss_ar = []
    for i in range(config["n_runs"]):
        print(config)
        pl.seed_everything(i)
        val_loss = run_game_with_config(config)[config["metric"]]
        loss_ar.append(val_loss)

    loss = np.mean(loss_ar)

    if loss < lowest_loss:
        lowest_loss = loss
        best = config

print(lowest_loss)
print(best)

print_best_pretty(config, best)




###Config (Move to some file or something for easy training and experimentiation
