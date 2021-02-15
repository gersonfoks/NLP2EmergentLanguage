import torch
import pytorch_lightning as pl

import argparse
import yaml

from callbacks.msg_callback import MsgCallback, MsgFrequencyCallback, EntropyMeasure, MeasureCallbacks, \
    ResetDatasetCallback, MsgLength, DistinctSymbolMeasure
from models.pl_model import SignallingGameModel
from utils import get_sender, get_receiver, get_shape_signalling_game, get_predictor, cross_entropy_loss_2

parser = argparse.ArgumentParser(description='Run an experiment defined an a yml file')

parser.add_argument('--config', default="config/example_config.yml", required=False)

args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f)

### We first load the datasets

pl.seed_everything(config["seed"])

samples_per_epoch_train = config['samples_per_epoch_train']
samples_per_epoch_test = config['samples_per_epoch_test']

max_epochs = config["max_epochs"]

msg_len = config["msg_len"]
n_symbols = config["n_symbols"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrain = config["pretrain"]

sender = get_sender(n_symbols, msg_len, device, fixed_size=config["fixed_size"], pretrain=pretrain,
                    pretrain_n_epochs=config["pretrain_n_epochs"])
receiver = get_receiver(n_symbols, msg_len, device, pretrain=pretrain, pretrain_n_epochs=config["pretrain_n_epochs"])

train_dataloader, test_dataloader = get_shape_signalling_game(batch_size=config["batch_size"],
                                                              samples_per_epoch_train=samples_per_epoch_train,
                                                              samples_per_epoch_test=samples_per_epoch_test)

predictor = get_predictor(n_symbols, 128, device)

loss_module = torch.nn.CrossEntropyLoss()

loss_module_predictor = cross_entropy_loss_2

signalling_game_model = SignallingGameModel(sender, receiver, loss_module, predictor=predictor,
                                            loss_module_predictor=loss_module_predictor, hparams=config).to(device)

to_sample_from = next(iter(test_dataloader))[:5]

msg_callback = MsgCallback(to_sample_from, )

freq_callback = MsgFrequencyCallback(to_sample_from)

### We create all the measures
symbol_entropy = EntropyMeasure('symbol entropy')
bi_gram_entropy = EntropyMeasure('bigram entropy', n_gram=2)
len_measure = MsgLength('avg msg length', stop_symbol=config['n_symbols'] - 1)
distinct_symbol_measure = DistinctSymbolMeasure('Number of distinct symbols')

measure_callbacks = MeasureCallbacks(test_dataloader, measures=[symbol_entropy, bi_gram_entropy, len_measure, distinct_symbol_measure])

reset_trainer = ResetDatasetCallback(train_dataloader.dataset)

trainer = pl.Trainer(default_root_dir='logs',
                     checkpoint_callback=False,
                     # checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                     gpus=1 if torch.cuda.is_available() else 0,
                     max_epochs=max_epochs,
                     log_every_n_steps=1,
                     callbacks=[msg_callback, freq_callback, measure_callbacks, reset_trainer],
                     progress_bar_refresh_rate=1,
                     )
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

### Save in the list the experiment number and the experiment


trainer.fit(signalling_game_model, train_dataloader)
