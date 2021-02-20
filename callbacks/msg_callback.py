from collections import Counter

import pytorch_lightning as pl
import numpy as np
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
            msg, msg_packed, out, out_probs, prediction_logits, prediction_probs = pl_module.forward(self.receiver_imgs, choices)

            logger = trainer.logger.experiment

            text = self.msg_to_text(msg)

            logger.add_text('msgs', text, trainer.current_epoch)

    def msg_to_text(self, msg):

        indices = torch.argmax(msg.permute(1,0,2), dim=-1)

        indices_numpy = indices.cpu().numpy()

        text_list = [' '.join([str(j) for j in list(i)]) for i in list(indices_numpy)]

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
            msg, msg_packed, out, out_probs, prediction_logits, prediction_probs = pl_module.forward(self.receiver_imgs, choices)

            logger = trainer.logger.experiment

            freq = self.msg_to_freq(msg)

            logger.add_text('freqs', freq, trainer.current_epoch)

    def msg_to_freq(self, msg):
        indices = list(torch.argmax(msg, dim=-1).flatten().cpu().numpy())
        counter = Counter(indices)

        return str([(key, value) for key, value in sorted(counter.items())])


class MeasureCallbacks(pl.Callback):
    '''
    Creates a plot based around a digit
    '''

    def __init__(self, dataloader, measures, every_n_epochs=1):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.dataloader = dataloader
        self.measures = measures

    @torch.no_grad()
    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """

        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:

            ## We generate all the messages
            msgs = []

            for sender_imgs, receiver_imgs, target in self.dataloader:
                sender_imgs = sender_imgs.to(pl_module.device)
                receiver_imgs = [receiver_img.to(pl_module.device) for receiver_img in receiver_imgs]

                msg, msg_packed, out, out_probs, prediction_logits, prediction_probs = pl_module.forward(sender_imgs, receiver_imgs)

                #Make batch first
                msg = torch.argmax(msg, dim=-1).permute(1,0)
                msgs.append(msg)



            msgs = torch.cat(msgs)
            logger = trainer.logger.experiment
            for measure in self.measures:
                m = measure.make_measure(msgs)
                logger.add_scalar(measure.name, m, trainer.current_epoch)


class Measure:

    def __init__(self, name):
        self.name = name

    def make_measure(self, msgs):
        pass


class EntropyMeasure(Measure):

    def __init__(self, name, stop_symbol=None, n_gram=1):
        '''

        :param name: name of the measure
        :param fixed_length_message: The stop sign that is used
        :param n_gram: how big the ngrams should be.
        '''
        super().__init__(name)
        self.stop_symbol = stop_symbol
        self.n_gram = n_gram

    def make_measure(self, msgs):

        n_grams = self.create_n_grams(msgs)

        count = [val for key, val in Counter(n_grams).items()]

        total = sum(count)

        percentages = [c / total for c in count]

        return -sum([p * np.log2(p) for p in percentages])

    def create_n_grams(self, msgs):
        if self.n_gram == 1:
            msgs = msgs.flatten()
            msgs = list(msgs.cpu().numpy())
            return msgs

        msgs_list = list(msgs.cpu().numpy())
        result = []
        if self.n_gram > 1:
            l = len(msgs_list[0])
            for msg in msgs_list:
                for i in range(l - self.n_gram):
                    result.append(tuple(msg[i:i + self.n_gram]))

        return result


class MsgLength(Measure):

    def __init__(self, name, stop_symbol=None):
        '''

        :param name: name of the measure
        :param fixed_length_message: The stop sign that is used
        :param n_gram: how big the ngrams should be.
        '''
        super().__init__(name)
        self.stop_symbol = stop_symbol

    def make_measure(self, msgs):
        mask = ~msgs.ge(self.stop_symbol)

        msgs = torch.masked_select(msgs, mask)

        total_len = len(msgs)
        n_msgs = mask.shape[0]

        return 1 + total_len / n_msgs


class DistinctSymbolMeasure(Measure):

    def __init__(self, name):
        '''

        :param name: name of the measure
        :param fixed_length_message: The stop sign that is used
        :param n_gram: how big the ngrams should be.
        '''
        super().__init__(name)

    def make_measure(self, msgs):
        return len(torch.unique(msgs))


class ResetDatasetCallback(pl.Callback):

    def __init__(self, dataset, every_n_epochs=1):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.dataset = dataset

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            self.dataset.reset()
