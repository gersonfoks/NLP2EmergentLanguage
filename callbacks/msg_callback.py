from collections import Counter

import pytorch_lightning as pl
import torch


class MsgCallback(pl.Callback):
    '''
    Creates a plot based around a digit
    '''

    def __init__(self, to_sample_from, n_samples=5, every_n_epochs=1, save_to_disk=False, ):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.to_sample_from = to_sample_from
        self.n_samples = n_samples
        self.save_to_disk = save_to_disk

        self.receiver_imgs = to_sample_from[0]
        self.sender_choices = to_sample_from[1]

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.receiver_imgs = self.receiver_imgs.to(pl_module.device)
            choices = []
            for choice in self.sender_choices:
                choices.append(choice.to(pl_module.device))
            msg, out, out_probs, prediction_logits, prediction_probs = pl_module.forward(self.receiver_imgs, choices)

            logger = trainer.logger.experiment

            text = self.msg_to_text(msg)

            logger.add_text('msgs', text, trainer.current_epoch)

    def msg_to_text(self, msg):
        indices = torch.argmax(msg, dim=-1)

        indices_numpy = indices.cpu().numpy()

        text_list = [' '.join([str(j) for j in  list(i)]) for i in list(indices_numpy)]

        text = '|'.join(text_list)
        return text


class MsgFrequencyCallback(pl.Callback):
    '''
    Creates a plot based around a digit
    '''

    def __init__(self, to_sample_from, n_samples=5, every_n_epochs=1, save_to_disk=False, ):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.to_sample_from = to_sample_from
        self.n_samples = n_samples
        self.save_to_disk = save_to_disk

        self.receiver_imgs = to_sample_from[0]
        self.sender_choices = to_sample_from[1]

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.receiver_imgs = self.receiver_imgs.to(pl_module.device)
            choices = []
            for choice in self.sender_choices:
                choices.append(choice.to(pl_module.device))
            msg, out, out_probs, prediction_logits, prediction_probs = pl_module.forward(self.receiver_imgs, choices)

            logger = trainer.logger.experiment

            freq = self.msg_to_freq(msg)

            logger.add_text('freqs', freq, trainer.current_epoch)

    def msg_to_freq(self, msg):
        indices = list(torch.argmax(msg, dim=-1).flatten().cpu().numpy())
        counter = Counter(indices)

        return str([(key, value) for key, value in sorted(counter.items())])


