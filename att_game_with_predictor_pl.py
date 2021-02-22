import torch
import pytorch_lightning as pl

from attribute_game.pl_model import AttributeBaseLineModel, AttributeModelWithPrediction
from attribute_game.utils import get_sender, get_receiver, get_predictor
from callbacks.msg_callback import MsgCallback, MsgFrequencyCallback, EntropyMeasure, MeasureCallbacks, \
    ResetDatasetCallback, DistinctSymbolMeasure

### Set to a number for faster prototyping
from datasets.AttributeDataset import get_attribute_game
from utils import cross_entropy_loss_2

batch_size = 32
n_attributes = 4
attributes_size = 3

n_receiver = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

samples_per_epoch_train = int(10e3)
samples_per_epoch_test = int(10e3)
max_epochs = 20

hidden_size_predictor = 128

msg_len = 5
n_symbols = 25

fixed_size = False
pack_message = not fixed_size

pretrain_n_epochs = 3

hparams = {'learning_rate': 0.001, "predictor_loss_weight": 0.01}

sender = get_sender(n_attributes, attributes_size, n_symbols, msg_len, device, fixed_size=fixed_size,
                    pretrain_n_epochs=pretrain_n_epochs)
receiver = get_receiver(n_attributes, attributes_size, n_receiver, n_symbols, msg_len, device, fixed_size=fixed_size,
                        pretrain_n_epochs=pretrain_n_epochs)

predictor = get_predictor(n_symbols, hidden_size_predictor, device)

train_dataloader, test_dataloader = get_attribute_game(n_attributes, attributes_size,n_receiver=n_receiver,
                                                       samples_per_epoch_train=samples_per_epoch_train,
                                                       samples_per_epoch_test=samples_per_epoch_test, batch_size=batch_size)

loss_module = torch.nn.CrossEntropyLoss()
loss_module_predictor = cross_entropy_loss_2

signalling_game_model = AttributeModelWithPrediction(sender, receiver, loss_module, predictor, loss_module_predictor,
                                                     hparams=hparams, pack_message=pack_message).to(device)

to_sample_from = next(iter(test_dataloader))[:5]

msg_callback = MsgCallback(to_sample_from, )

freq_callback = MsgFrequencyCallback(to_sample_from)

### We create all the measures
symbol_entropy = EntropyMeasure('symbol entropy')
bi_gram_entropy = EntropyMeasure('bigram entropy', n_gram=2)
distinct_measure = DistinctSymbolMeasure('Distinct Symbols')
measure_callbacks = MeasureCallbacks(test_dataloader, measures=[symbol_entropy, bi_gram_entropy, distinct_measure])

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

trainer.fit(signalling_game_model, train_dataloader, train_dataloader)



