from argparse import ArgumentParser
from datetime import datetime
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
import numpy as np

class BaseModel(pl.LightningModule):
    def __init__(self, encoder, decoder, hparams, ):
        super().__init__()
        self.hparams = hparams
        #self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder

        self.epoch_count = 0
        self.decrease_every = hparams.decrease_every
        self.decrease_by = hparams.decrease_by
        self.decrease_min = hparams.decrease_min

        self.shuffle_beta = hparams.shuffle_beta
        self.shuffle_theta = hparams.shuffle_theta

        self.cosine_sim = nn.CosineSimilarity(dim=-3)
        self.MSE_loss = nn.MSELoss()

    def forward(self, batch):
        # ENCODING
        scans_shapes = batch['scans'].shape
        # reshape the scans
        # from [batch_size, n_slices, n_mod, channels, H, W]
        # to   [temp_batch_size, channels, H, W]
        scans = batch['scans'].reshape([-1] + list(scans_shapes[3:]))
        beta, theta = self.encoder(scans)
        beta = torch.argmax(beta, dim=1)
        beta = F.one_hot(beta)
        beta = torch.moveaxis(beta, -1, 1)
        # what decoding do we want here?


    def step(self, batch):
        """

        :param batch:       dict containing two tensors.
                                first tensor: scans [batch_size, n_slices, n_mod, channels, H, W]
                                second tensor: mod: [batch_size, n_slices, n_mod]
        :param batch_idx:   compatibility
        :return:
        """
        # ENCODING
        scans_shapes = batch['scans'].shape
        # reshape the scans
        # from [batch_size, n_slices, n_mod, channels, H, W]
        # to   [temp_batch_size, channels, H, W]
        scans = batch['scans'].reshape([-1] + list(scans_shapes[3:]))
        # step one: shuffle the slices & modalities
        slices_idx = np.arange(scans.shape[0])
        np.random.shuffle(slices_idx)
        scans = scans[slices_idx]
        # pass through encoder
        beta, theta = self.encoder(scans)
        beta = self.encoder.GumbelSoftmax(beta)
        # unshuffle for mental sanity
        beta = beta[np.argsort(slices_idx)]
        theta = theta[np.argsort(slices_idx)]

        beta = beta.reshape(scans_shapes[:3] + beta.shape[1:])
        theta = theta.reshape(list(scans_shapes[:3]) + [theta.shape[-1]])
        similarity_loss = -torch.mean(self.cosine_sim(beta[:, :, 0], beta[:, :, 1]))

        # SHUFFLE
        reconstruction_loss = torch.empty(0, requires_grad=True).sum()
        reconstruction_loss = reconstruction_loss.to(beta.device)
        for _ in range(4):
            # shuffle beta
            if self.shuffle_beta:
                temp_beta = self.randomise_beta(beta)
            else:
                temp_beta = beta.copy()
            # shuffle theta
            if self.shuffle_theta:
                temp_theta, temp_scans = self.randomise_theta(theta, batch['scans'])
            else:
                temp_theta = theta.copy()
                temp_scans = batch['scans'].copy()
            # reshape
            temp_beta = temp_beta.reshape([-1] + list(beta.shape[3:]))
            temp_scans = temp_scans.reshape([-1] + list(scans_shapes[3:]))
            temp_theta = temp_theta.reshape([-1] + [theta.shape[-1]])

            # DECODING
            predicted_scans = self.decoder(temp_beta, temp_theta)
            reconstruction_loss = reconstruction_loss + self.MSE_loss(predicted_scans, temp_scans)
        reconstruction_loss = reconstruction_loss / 4.
        loss = reconstruction_loss + self.hparams.lambda_sim * similarity_loss
        return {'loss': loss,
                'rec_loss': reconstruction_loss,
                'sim_loss': similarity_loss}


    def training_step(self, batch, batch_idx):
        _dict = self.step(batch)
        for key in _dict:
            self.log('train_' + key, _dict[key], on_epoch=True)
        return _dict

    def validation_step(self, batch, batch_idx):
        _dict = self.step(batch)
        for key in _dict:
            self.log('valid_' + key, _dict[key], on_epoch=True)
        return _dict

    def test_step(self, batch, batch_idx):
        _dict = self.step(batch)
        for key in _dict:
            self.log('test_' + key, _dict[key], on_epoch=True)
        return _dict

    def training_epoch_end(self, outputs):
        # decrease the gumbel-softmax temperature
        self.epoch_count += 1
        if self.epoch_count > self.decrease_every:
            self.encoder.temperature -= self.decrease_by
            self.encoder.temperature = min(self.decrease_min, self.encoder.temperature)

    #def training_epoch_end(self, training_step_outputs):

    #def validation_epoch_end(self, validation_step_outputs):

    #def test_epoch_end(self, test_step_outputs):

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        #return torch.optim.Adam(nn.ParameterList((self.encoder.parameters, self.decoder.parameters())),
        #        lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

    def load_from_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict']
        self.load_state_dict(state_dict)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--lambda_sim', type=float, default=0.0005)
        parser.add_argument('--shuffle_beta', type=bool, default=True)
        parser.add_argument('--shuffle_theta', type=bool, default=True)
        return parser

    @property
    def time(self) -> str:
        return str(datetime.now())[:10]+'--'+'-'.join(str(datetime.now())[11:-7].rsplit(':'))

    @staticmethod
    def randomise_beta(beta): # todo: use moveaxis instead of swapaxes
        # torch.where operates on the last dim, hence we need to swap axes (because I couldnt find rotation)
        # from [batch_size, n_slices, n_mod, beta_channels, H, W]
        # to   [batch_size, n_slices, n_mod, W, H, beta_channels]
        # to   [batch_size, H, n_mod, W, n_slices, beta_channels]
        # to   [W, H, n_mod, batch_size, n_slices, beta_channels]
        beta = torch.swapaxes(beta, -3, -1)
        beta = torch.swapaxes(beta, 1, -2)
        beta = torch.swapaxes(beta, 0, -3)
        random_selection1 = torch.randint(beta.shape[2], beta.shape[-3:])
        random_selection1 = random_selection1.to(beta.device)
        # to   [W, H, batch_size, n_slices, beta_channels]
        mixed_beta1 = torch.where(random_selection1 == 0, beta[:, :, 0], beta[:, :, 1])
        # to   [W, H, batch_size, n_slices, beta_channels]
        # to   [batch_size, H, W, n_slices, beta_channels]
        # to   [batch_size, n_slices, W, H, beta_channels]
        # to   [batch_size, n_slices, beta_channels, H, W]
        mixed_beta1 = torch.swapaxes(mixed_beta1, 0, -3)
        mixed_beta1 = torch.swapaxes(mixed_beta1, 1, -2)
        mixed_beta1 = torch.swapaxes(mixed_beta1, -3, -1)
        #
        random_selection2 = torch.randint(beta.shape[2], beta.shape[-3:])
        random_selection2 = random_selection2.to(beta.device)
        mixed_beta2 = torch.where(random_selection2 == 0, beta[:, :, 0], beta[:, :, 1])
        mixed_beta2 = torch.swapaxes(mixed_beta2, 0, -3)
        mixed_beta2 = torch.swapaxes(mixed_beta2, 1, -2)
        mixed_beta2 = torch.swapaxes(mixed_beta2, -3, -1)


        # to   [batch_size, n_slices, n_mod (mixed), beta_channels, H, W]
        mixed_beta = torch.stack((mixed_beta1, mixed_beta2), dim=2)
        return mixed_beta

    @staticmethod
    def randomise_theta(theta, scans):
        # torch.where operates on the last dim, hence we need to swap axes (because I couldnt find rotation)
        # from [batch_size, n_slices, n_mod, theta_channels]
        # to   [n_mod, n_slices, batch_size, theta_channels]
        # to   [n_mod, theta_channels, batch_size, n_slices]
        theta = torch.swapaxes(theta, 0, 2)
        theta = torch.swapaxes(theta, 1, 3)
        # from [batch_size, n_slices, n_mod, channels, H, W]
        # to   [batch_size, W, n_mod, channels, H, n_slices]
        # to   [H, W, n_mod, channels, batch_size, n_slices]
        scans = torch.swapaxes(scans, 1, -1)
        scans = torch.swapaxes(scans, 0, -2)

        random_selection = torch.randint(theta.shape[0], theta.shape[-2:])
        random_selection = random_selection.to(theta.device)
        # to   [theta_channels, batch_size, n_slices]
        # to   [batch_size, theta_channels, n_slices]
        # to   [batch_size, n_slices, theta_channels]
        mixed_theta = torch.where(random_selection == 0, theta[0], theta[1])
        mixed_theta = torch.swapaxes(mixed_theta, 0, 1)
        mixed_theta1 = torch.swapaxes(mixed_theta, 1, 2)
        # to   [H, W, channels, batch_size, n_slices]
        # to   [batch_size, W, channels, H, n_slices]
        # to   [batch_size, n_slices, channels, H, W]
        mixed_scans = torch.where(random_selection == 0, scans[:, :, 0], scans[:, :, 1])
        mixed_scans = torch.swapaxes(mixed_scans, 0, -2)
        mixed_scans1 = torch.swapaxes(mixed_scans, 1, -1)
        #
        random_selection = torch.randint(theta.shape[0], theta.shape[-2:])
        random_selection = random_selection.to(theta.device)
        # to   [theta_channels, batch_size, n_slices]
        # to   [batch_size, theta_channels, n_slices]
        # to   [batch_size, n_slices, theta_channels]
        mixed_theta = torch.where(random_selection == 0, theta[0], theta[1])
        mixed_theta = torch.swapaxes(mixed_theta, 0, 1)
        mixed_theta2 = torch.swapaxes(mixed_theta, 1, 2)
        # to   [H, W, channels, batch_size, n_slices]
        # to   [batch_size, W, channels, H, n_slices]
        # to   [batch_size, n_slices, channels, H, W]
        mixed_scans = torch.where(random_selection == 0, scans[:, :, 0], scans[:, :, 1])
        mixed_scans = torch.swapaxes(mixed_scans, 0, -2)
        mixed_scans2 = torch.swapaxes(mixed_scans, 1, -1)

        # to   [batch_size, n_slices, n_mod (mixed), beta_channels, H, W]
        mixed_theta = torch.stack((mixed_theta1, mixed_theta2), dim=2)
        mixed_scans = torch.stack((mixed_scans1, mixed_scans2), dim=2)
        return mixed_theta, mixed_scans


