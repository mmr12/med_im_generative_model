from argparse import ArgumentParser
from datetime import datetime
import pandas as pd
import itertools
from collections import OrderedDict
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
import numpy as np

class BaseModel(pl.LightningModule):
    def __init__(self, beta_encoder, theta_encoder, decoder, adv_net, hparams, ):
        super().__init__()
        self.hparams = hparams
        #self.save_hyperparameters()
        self.beta_encoder = beta_encoder
        self.theta_encoder = theta_encoder
        self.decoder = decoder
        self.adv_net = adv_net

        self.epoch_count = 0
        self.decrease_every = hparams.decrease_every
        self.decrease_by = hparams.decrease_by
        self.decrease_min = hparams.decrease_min

        self.MAE_loss = nn.L1Loss()
        self.adv_loss = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        # ENCODING
        scans_shapes = batch['scans'].shape
        # reshape the scans
        # from [batch_size, n_slices, n_mod, channels, H, W]
        # to   [temp_batch_size, channels, H, W]
        scans = batch['scans'].reshape([-1] + list(scans_shapes[3:]))
        beta = self.beta_encoder(scans)
        beta = torch.argmax(beta, dim=1)
        beta = F.one_hot(beta)
        beta = torch.swapaxes(beta, -1, 1)

        theta = self.theta_encoder(scans)
        # what decoding do we want here?

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, batch, scans_shapes):
        # ENCODING

        # reshape the scans
        # from [batch_size, n_slices, n_mod, channels, H, W]
        # to   [temp_batch_size, channels, H, W]
        scans = batch['scans'].reshape([-1] + list(scans_shapes[3:]))
        # step one: shuffle the slices & modalities
        slices_idx = torch.randperm(scans.shape[0])
        scans = scans[slices_idx]
        # pass through encoder
        # encode beta
        beta = self.beta_encoder(scans)
        beta_adv = beta.detach()
        beta = self.beta_encoder.GumbelSoftmax(beta)
        # encode theta through CVAE
        mu, log_var = self.theta_encoder(scans)
        theta = self.sample(mu, log_var)
        # unshuffle for mental sanity
        beta = beta[torch.argsort(slices_idx)]
        theta = theta[torch.argsort(slices_idx)]
        return beta, beta_adv, theta, mu, log_var

    def discriminator_step(self, batch, beta_adv):
        n_classes = beta_adv.shape[1]
        beta_adv = torch.argmax(beta_adv, dim=1)
        beta_adv = F.one_hot(beta_adv, n_classes)
        beta_adv = torch.swapaxes(beta_adv, -1, 1).type(torch.float)
        # shuffle
        shuffle_idx = torch.randperm(beta_adv.shape[0])
        beta_adv = beta_adv[shuffle_idx]
        site_label = batch['site'].reshape(-1)[shuffle_idx]
        loss = self.adv_loss(self.adv_net(beta_adv), site_label.type(torch.long))
        return loss

    def latent_losses_step(self, beta, mu, log_var, scans_shapes):
        # LATENT REP LOSSES
        # beta L1 loss
        beta = beta.reshape(scans_shapes[:3] + beta.shape[1:])
        similarity_loss = torch.mean(torch.abs(beta[:, :, 0] - beta[:, :, 1]))
        # theta KL loss
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return similarity_loss, kl_loss

    def adversarial_loss_step(self, batch, beta_adv):
        # optimise against adversarial loss
        beta_adv = torch.argmax(beta_adv, dim=1)
        beta_adv = F.one_hot(beta_adv)
        beta_adv = torch.swapaxes(beta_adv, -1, 1).type(torch.float)
        # shuffle
        shuffle_idx = torch.randperm(beta_adv.shape[0])
        beta_adv = beta_adv[shuffle_idx]
        site_label = batch['site'].reshape(-1)[shuffle_idx]
        loss = self.adv_loss(self.adv_net(beta_adv), (1-site_label).type(torch.long))
        return loss

    def reconstruction_loss_step(self, batch, beta, theta, scans_shapes):
        beta = beta.reshape(scans_shapes[:3] + beta.shape[1:])
        theta = theta.reshape(list(scans_shapes[:3]) + [theta.shape[-1]])
        #
        reconstruction_loss = torch.empty(0, requires_grad=True).sum()
        reconstruction_loss = reconstruction_loss.to(beta.device)
        for _ in range(4):
            # shuffle beta
            temp_beta = self.randomise_beta(beta)
            # shuffle theta
            temp_theta, temp_scans = self.randomise_theta(theta, batch['scans'])
            # reshape
            temp_beta = temp_beta.reshape([-1] + list(beta.shape[3:]))
            temp_scans = temp_scans.reshape([-1] + list(scans_shapes[3:]))
            temp_theta = temp_theta.reshape([-1] + [theta.shape[-1]])

            # DECODING
            predicted_scans = self.decoder(temp_beta, temp_theta)
            reconstruction_loss = reconstruction_loss + self.MAE_loss(predicted_scans, temp_scans)
        reconstruction_loss = reconstruction_loss / 4.
        return reconstruction_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        """

        :param batch:       dict containing two tensors.
                                first tensor: scans [batch_size, n_slices, n_mod, channels, H, W]
                                second tensor: mod: [batch_size, n_slices, n_mod]
        :param batch_idx:   compatibility
        :return:
        """
        scans_shapes = batch['scans'].shape

        beta, beta_adv, theta, mu, log_var = self.encode(batch, scans_shapes)

        # TRAIN DISCRIMINATOR
        if optimizer_idx == 0:
            discr_loss = self.discriminator_step(batch, beta_adv)
            tqdm_dict = {'discr_loss': discr_loss}
            tqdm_dict = {'train_'+key: tqdm_dict[key] for key in tqdm_dict}
            output = OrderedDict({
                'loss': discr_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            self.log_dict(tqdm_dict, on_epoch=True, on_step=False)
            return output

        # TRAIN GENERATOR
        if optimizer_idx == 1:
            similarity_loss, kl_loss = self.latent_losses_step(beta, mu, log_var, scans_shapes)
            adv_loss = self.adversarial_loss_step(batch, beta)
            reconstruction_loss = self.reconstruction_loss_step(batch, beta, theta, scans_shapes)

            loss = reconstruction_loss \
                   + self.hparams.lambda_sim * similarity_loss \
                   + self.hparams.lambda_adv * adv_loss \
                   + self.hparams.lambda_kl * kl_loss
            tqdm_dict = {'recon_loss': reconstruction_loss,
                         'adv_loss': adv_loss,
                         'sim_loss': similarity_loss,
                         'kl_loss': kl_loss,
                         'non_discr_tot_loss': loss}
            tqdm_dict = {'train_' + key: tqdm_dict[key] for key in tqdm_dict}
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output


    def validation_step(self, batch, batch_idx):
        scans_shapes = batch['scans'].shape
        beta, beta_adv, theta, p, q = self.encode(batch, scans_shapes)
        discr_loss = self.discriminator_step(batch, beta_adv)
        similarity_loss, kl_loss = self.latent_losses_step(beta, theta, p, q, scans_shapes)
        adv_loss = self.adversarial_loss_step(batch, beta)
        reconstruction_loss = self.reconstruction_loss_step(batch, beta, theta, scans_shapes)
        loss = reconstruction_loss \
               + self.hparams.lambda_sim * similarity_loss \
               + self.hparams.lambda_adv * adv_loss \
               + self.hparams.lambda_kl * kl_loss
        tqdm_dict = {'discr_loss': discr_loss,
                     'recon_loss': reconstruction_loss,
                     'adv_loss': adv_loss,
                     'sim_loss': similarity_loss,
                     'kl_loss': kl_loss,
                     'non_discr_tot_loss': loss}
        tqdm_dict = {'val_' + key: tqdm_dict[key] for key in tqdm_dict}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def test_step(self, batch, batch_idx):
        scans_shapes = batch['scans'].shape
        beta, beta_adv, theta, p, q = self.encode(batch, scans_shapes)
        discr_loss = self.discriminator_step(batch, beta_adv)
        similarity_loss, kl_loss = self.latent_losses_step(beta, theta, p, q, scans_shapes)
        adv_loss = self.adversarial_loss_step(batch, beta)
        reconstruction_loss = self.reconstruction_loss_step(batch, beta, theta, scans_shapes)
        loss = reconstruction_loss \
               + self.hparams.lambda_sim * similarity_loss \
               + self.hparams.lambda_adv * adv_loss \
               + self.hparams.lambda_kl * kl_loss
        tqdm_dict = {'discr_loss': discr_loss,
                     'recon_loss': reconstruction_loss,
                     'adv_loss': adv_loss,
                     'sim_loss': similarity_loss,
                     'kl_loss': kl_loss,
                     'non_discr_tot_loss': loss}
        tqdm_dict = {'test_' + key: tqdm_dict[key] for key in tqdm_dict}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def training_epoch_end(self, outputs):
        # decrease the gumbel-softmax temperature
        self.epoch_count += 1
        if self.epoch_count > self.decrease_every:
            self.encoder.temperature -= self.decrease_by
            self.encoder.temperature = min(self.decrease_min, self.encoder.temperature)

    def configure_optimizers(self):
        adv_params = self.adv_net.parameters()
        CVAE_params = list(self.beta_encoder.parameters()) + list(self.theta_encoder.parameters())  + list(self.decoder.parameters())
        lr = self.hparams.learning_rate
        l2 = self.hparams.weight_decay

        opt_adv = torch.optim.Adam(adv_params, lr=lr, weight_decay=l2)
        opt_CVAE = torch.optim.Adam(CVAE_params, lr=lr, weight_decay=l2)
        return [opt_adv, opt_CVAE], []

    def load_from_checkpoint(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict']
        self.load_state_dict(state_dict)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0)
        parser.add_argument('--lambda_sim', type=float, default=0.0005)
        parser.add_argument('--lambda_adv', type=float, default=0.0005)
        parser.add_argument('--lambda_kl', type=float, default=0.0005)
        return parser

    @property
    def time(self) -> str:
        return str(datetime.now())[:10]+'--'+'-'.join(str(datetime.now())[11:-7].rsplit(':'))

    @staticmethod
    def randomise_beta(beta):
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


