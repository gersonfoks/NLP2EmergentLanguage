import pytorch_lightning as pl
import torch


class SignallingGameModel(pl.LightningModule):

    def __init__(self, sender, receiver, loss_module_receiver, predictor=None, loss_module_predictor=None):
        super().__init__()
        self.sender = sender
        self.receiver = receiver
        self.predictor = predictor
        self.loss_module_receiver = loss_module_receiver
        self.loss_module_predictor = loss_module_predictor

    def forward(self, sender_img, receiver_choices):
        msg = self.sender(sender_img)
        ##Make an all zeros msg to test if we are not just remembering the dataset.
        # msg = torch.zeros((len(msg), 5, 3)).to(self.device)
        prediction_logits, prediction_probs = None, None

        if self.predictor:
            start_symbols = torch.zeros(len(sender_img), 1, self.sender.n_symbols).to(self.device)
            msgs = torch.cat([start_symbols, msg], dim=1)[:]
            msg_in = msgs[:, :-1]
            prediction_logits, prediction_probs, hidden = self.predictor(msg_in)

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

            prediction_squeezed = prediction_logits.reshape(-1, self.sender.n_symbols)
            prediction_probs = prediction_probs.reshape(-1, self.sender.n_symbols)
            msg_target = msg.reshape(-1, self.sender.n_symbols)

            loss_predictor = self.loss_module_predictor(prediction_squeezed, msg_target)
            indices = torch.argmax(msg_target, dim=-1)
            accuracyPredictions = torch.argmax(prediction_probs, dim=-1)

            correct = (accuracyPredictions == indices).sum().item()
            predictor_accuracy = correct / len(indices)
            ### Log the accuracy
            self.log("accuracy predictor", predictor_accuracy, on_step=True, on_epoch=True)

        loss_receiver = self.loss_module_receiver(out_probs, target)

        loss = loss_receiver + 0.001 * loss_predictor

        predicted_indices = torch.argmax(out_probs, dim=-1)

        correct = (predicted_indices == target).sum().item() / batch_size

        self.log("loss_predictor", loss_predictor, on_step=True, on_epoch=True)
        self.log("loss_receiver", loss_receiver, on_step=True, on_epoch=True)
        self.log("total_loss", loss, on_step=True, on_epoch=True)
        self.log("accuracy", correct, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        parameters = list(self.sender.parameters()) + list(self.receiver.parameters())
        if self.predictor:
            parameters += list(self.predictor.parameters())
        optimizer = torch.optim.Adam(
            parameters,
            lr=0.001)
        return optimizer
