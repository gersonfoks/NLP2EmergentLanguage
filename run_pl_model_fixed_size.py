import torch
import pytorch_lightning as pl

from callbacks.msg_callback import MsgCallback, MsgFrequencyCallback
from pl_model import SignallingGameModel
from utils import get_mnist_signalling_game, get_sender, get_receiver, get_predictor

###Config (Move to some file or something for easy training and experimentiation


### Set to a number for faster prototyping
size= 100

msg_len = 5
n_symbols = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pl.seed_everything(42)

sender = get_sender(  n_symbols, msg_len, device)
receiver = get_receiver(n_symbols, msg_len, device)

train_dataloader, test_dataloader = get_mnist_signalling_game(size=size)

sender = get_sender(n_symbols, msg_len, device)
receiver = get_receiver(n_symbols, msg_len, device)

###Todo make 128 variable
predictor = get_predictor(n_symbols, 128, device)

loss_module = torch.nn.CrossEntropyLoss()

loss_module_predictor = torch.nn.MSELoss()

signalling_game_model = SignallingGameModel(sender, receiver, loss_module, predictor=predictor, loss_module_predictor=loss_module_predictor).to(device)



to_sample_from = next(iter(test_dataloader))[:5]


msg_callback = MsgCallback(to_sample_from, )

freq_callback = MsgFrequencyCallback(to_sample_from)

trainer = pl.Trainer(default_root_dir='logs',
                     checkpoint_callback=False,
                     # checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_loss"),
                     gpus=1 if torch.cuda.is_available() else 0,
                     max_epochs=20,
                     log_every_n_steps=1,
                     callbacks=[msg_callback, freq_callback],
                     progress_bar_refresh_rate=1)
trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

trainer.fit(signalling_game_model, train_dataloader)
