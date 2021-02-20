import pytorch_lightning as pl
import torch


class AttributeBaseLineModel(pl.LightningModule):
    def __init__(self, sender, receiver, loss_module,
                 hparams=None):
        super().__init__()
        self.sender = sender
        self.receiver = receiver

        self.loss_module = loss_module

        self.hparams = hparams

    def forward(self, sender_img, receiver_choices):
        msg = self.sender(sender_img)

        out, out_probs = self.receiver(receiver_choices, msg)

        return msg, out, out_probs, None, None

    def training_step(self, batch, batch_idx):
        batch_size = len(batch[0])

        sender_img = batch[0].to(self.device)
        receiver_imgs = batch[1]
        target = batch[2].to(self.device)

        msg, out, out_probs, _, _ = self.forward(sender_img, receiver_imgs)

        loss = self.loss_module(out_probs, target)

        predicted_indices = torch.argmax(out_probs, dim=-1)

        correct = (predicted_indices == target).sum().item() / batch_size

        self.log("loss_receiver", loss, on_step=True, on_epoch=True)
        self.log("total_loss", loss, on_step=True, on_epoch=True)
        self.log("accuracy", correct, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        parameters = list(self.sender.parameters()) + list(self.receiver.parameters())

        optimizer = torch.optim.Adam(
            parameters,
            lr=self.hparams['learning_rate'])
        return optimizer
