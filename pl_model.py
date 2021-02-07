import pytorch_lightning as pl
import torch


class SignallingGameModel(pl.LightningModule):

    def __init__(self, sender, receiver, classifier, loss_module_receiver, loss_module_predictor, *args, **kwargs):
        super().__init__()
        self.sender = sender
        self.receiver = receiver
        self.predictor = classifier
        self.loss_module_receiver = loss_module_receiver
        self.loss_module_predictor = loss_module_predictor

    def forward(self, sender_img, receiver_choices):
        msg = self.sender(sender_img)

        prediction_logits, prediction_probs = None, None

        if self.predictor:
            prediction_logits, prediction_probs, hidden = self.predictor(msg)

        out, out_probs = self.receiver(receiver_choices, msg)

        return msg, out, out_probs, prediction_logits, prediction_probs

    def training_step(self, batch, batch_idx):

        batch_size = len(batch[0])

        sender_img = batch[0].to(self.device)
        receiver_imgs = batch[1]
        target = batch[2].to(self.device)

        msg, out, out_probs, prediction_logits, prediction_probs = self.forward(sender_img, receiver_imgs)


        loss_predictor = 0
        if self.predictor:
            prediction_squeazed = prediction_probs.reshape(-1, self.sender.n_symbols)

            msg_squezed = msg.reshape(-1, self.sender.n_symbols)

            loss_predictor = self.loss_module_predictor(prediction_squeazed, msg_squezed)

        loss_receiver = self.loss_module_receiver(out, target)

        loss = loss_receiver + 0.1 * loss_predictor

        predicted_indices = torch.argmax(out_probs, dim=-1)

        correct = (predicted_indices == target).sum().item()/batch_size

        ###Lekker loggen
        self.log("loss_predictor", loss_predictor, on_step=True, on_epoch=True)
        self.log("loss_receiver", loss_receiver, on_step=True, on_epoch=True)
        self.log("accuracy", correct, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        parameters = list(self.sender.parameters()) + list(self.receiver.parameters())
        if self.predictor:
            parameters += list(self.predictor.parameters())
        optimizer = torch.optim.Adam(
             parameters,
            lr=0.0001)
        return optimizer
