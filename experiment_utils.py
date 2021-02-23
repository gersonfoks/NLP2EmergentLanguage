from itertools import product

import torch
import yaml
import numpy as np
from attribute_game.pl_model import AttributeModelWithPrediction, AttributeBaseLineModel
from attribute_game.utils import get_sender, get_receiver, get_predictor
from callbacks.msg_callback import MsgCallback, MsgFrequencyCallback, EntropyMeasure, DistinctSymbolMeasure, \
    MeasureCallbacks, ResetDatasetCallback, MsgLength
from datasets.AttributeDataset import get_attribute_game
from utils import cross_entropy_loss_2
import pytorch_lightning as pl


def get_game(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_attributes = config["n_attributes"]
    attributes_size = config["attributes_size"]
    n_symbols = config["n_symbols"]
    msg_len = config["msg_len"]

    pretrain_n_epochs = config["pretrain_n_epochs"]
    fixed_size = config["fixed_size"]
    pack_message = not fixed_size
    n_receiver = config["n_receiver"]
    hparams = config
    loss_module = torch.nn.CrossEntropyLoss()
    pack_massage = not fixed_size
    sender = get_sender(n_attributes, attributes_size, n_symbols, msg_len, device, fixed_size=fixed_size,
                        pretrain_n_epochs=pretrain_n_epochs, )
    receiver = get_receiver(n_attributes, attributes_size, n_receiver, n_symbols, msg_len, device,
                            fixed_size=fixed_size,
                            pretrain_n_epochs=pretrain_n_epochs)

    if config["with_predictor"]:
        predictor = get_predictor(n_symbols, config["hidden_size_predictor"], device)
        loss_module_predictor = cross_entropy_loss_2
        signalling_game_model = AttributeModelWithPrediction(sender, receiver, loss_module, predictor,
                                                             loss_module_predictor,
                                                             hparams=hparams, pack_message=pack_message).to(device)
    else:
        signalling_game_model = AttributeBaseLineModel(sender, receiver, loss_module, hparams=hparams,
                                                       pack_message=pack_massage).to(device)

    return signalling_game_model


def run_game_with_config(config):
    n_attributes = config["n_attributes"]
    attributes_size = config["attributes_size"]

    samples_per_epoch_train = config["samples_per_epoch_train"]
    samples_per_epoch_test = config["samples_per_epoch_test"]
    pretrain_n_epochs = config["pretrain_n_epochs"]
    max_epochs = config["max_epochs"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, test_dataloader = get_attribute_game(n_attributes, attributes_size,
                                                           samples_per_epoch_train=samples_per_epoch_train,
                                                           samples_per_epoch_test=samples_per_epoch_test,
                                                           n_receiver=config["n_receiver"])

    signalling_game_model = get_game(config)

    to_sample_from = next(iter(test_dataloader))[:5]

    msg_callback = MsgCallback(to_sample_from, )

    freq_callback = MsgFrequencyCallback(to_sample_from)

    if config["fixed_size"]:
        stop_symbol = config["n_symbols"]
    else:
        stop_symbol = config["n_symbols"] - 1
    ### We create all the measures
    symbol_entropy = EntropyMeasure('symbol entropy', stop_symbol=stop_symbol)
    bi_gram_entropy = EntropyMeasure('bigram entropy', n_gram=2, stop_symbol=stop_symbol)
    tri_gram_entropy = EntropyMeasure('trigram entropy', n_gram=3, stop_symbol=stop_symbol)
    distinct_measure = DistinctSymbolMeasure('distinct symbols')


    msg_len_measure = MsgLength("msg_len", stop_symbol=stop_symbol)
    measure_callbacks = MeasureCallbacks(test_dataloader, measures=[symbol_entropy, bi_gram_entropy, distinct_measure, msg_len_measure, tri_gram_entropy])

    reset_trainer = ResetDatasetCallback(train_dataloader.dataset)

    trainer = pl.Trainer(default_root_dir='logs',
                         checkpoint_callback=False,
                         # checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=max_epochs,
                         log_every_n_steps=1,
                         callbacks=[msg_callback, freq_callback, measure_callbacks, reset_trainer],
                         progress_bar_refresh_rate=1)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    trainer.fit(signalling_game_model, train_dataloader, test_dataloader)

    return {**trainer.callback_metrics, **measure_callbacks.latest}


def print_best_pretty(base_config, best):
    names = [
        var[:-1] for var in base_config["grid_search_vars"]
    ]
    for name in names:
        print("{}, {}".format(name, best[name]))


def construct_configs(base_config):
    '''
    Construct configs
    '''
    to_combine = [
        list(base_config[var]) for var in base_config["grid_search_vars"]
    ]

    names = [
        var[:-1] for var in base_config["grid_search_vars"]
    ]

    vars = list(product(*to_combine))

    configs = []
    for vals in vars:
        c = base_config.copy()
        for i, name in enumerate(names):
            c[name] = vals[i]
        configs.append(c)
    return configs


def result_to_file(name, results):
    with open(name, 'w') as outfile:
        yaml.dump(results, outfile)


def get_summary_results(name):
    with open(name) as f:
        results = yaml.load(f)

    return {k: (np.mean(v), np.std(v)) for k, v in results.items()}


def create_name(config):
    return "experiment_{}_{}_{}_{}.yml".format(config["n_receiver"], config["n_attributes"], config["attributes_size"], config["with_predictor"])
