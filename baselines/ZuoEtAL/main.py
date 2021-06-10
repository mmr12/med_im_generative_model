from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
import sys, os
import torch
sys.path.append(os.getcwd())
from baselines.ZuoEtAL.LightningModule import BaseModel
from baselines.ZuoEtAL.model import BetaEncoder, ThetaEncoder, BetaDiscriminant, Decoder
from data_loaders.ixi import IXILoader

datasets = {'IXI': IXILoader}

def main():
    # -------
    # args
    # -------
    parser = ArgumentParser()
    # seed
    parser.add_argument('--seed', default=425, type=int)
    # data
    parser.add_argument('--data', default='IXI', type=str)
    parser.add_argument('--csv_path', default='/vol/biomedic2/mmr12/GitLab_projects/harmonisation/data/IXI_') #TODO: add full path
    parser.add_argument('--voxel_size', default=3, type=int)
    parser.add_argument('--n_slices', default=1, type=int)
    parser.add_argument('--modalities', default=('T1w', 'T2w'), type=tuple)
    parser.add_argument('--preload', default=True, type=bool)
    # model
    parser.add_argument('--out_channels', default=10, type=int)
    parser.add_argument('--unet_features', default=4, type=int)
    parser.add_argument('--beta_size', default=5, type=int)
    parser.add_argument('--theta_size', default=2, type=int)
    parser.add_argument('--kernel_size', default=5, type=int)
    parser.add_argument('--init_temperature', default=1, type=float)
    parser.add_argument('--decrease_every', default=5, type=int)
    parser.add_argument('--decrease_by', default=.1, type=float)
    parser.add_argument('--decrease_min', default=.5, type=float)
    # other stuff
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--test_only', default=False, type=bool)
    parser.add_argument('--preloaded_path', default='/vol/biomedic2/mmr12/GitLab_projects/harmonisation/data/IXI_4voxel_1slice_')

    # ------------
    # collect args
    # ------------
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BaseModel.add_model_specific_args(parser)
    args = parser.parse_args()
    # housekeeping
    torch.autograd.set_detect_anomaly(True)
    pl.seed_everything(args.seed)
    logger_path = 'lightning_logs/Zuo/{}/'.format(args.data)
    # GPU
    if args.gpus is not None and isinstance(args.gpus, int):
        # Make sure that it only uses a single GPU..
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
        args.gpus = 1

    # --------------
    # Load data
    # --------------
    data_files = {key: args.csv_path + key + '.csv' for key in ['train', 'eval', 'test']}
    if args.preloaded_path == 0:
        preloaded_paths = {key: None for key in ['train', 'eval', 'test']}
    else:
        preloaded_paths = {key: args.preloaded_path + key + '.pkl' for key in ['train', 'eval', 'test']}
    for key in data_files:
        if not os.path.isfile(data_files[key]):
            raise KeyError

    datasets = {key: IXILoader(data_files[key], args.voxel_size, args.n_slices, args.modalities, args.preload,
                               preloaded_paths[key]) for key in data_files}
    loaders = {key: DataLoader(datasets[key], batch_size=args.batch_size, num_workers=8) for key in datasets}

    # -------------
    # Load model
    # -------------
    beta_encoder = BetaEncoder(unet_init_features=args.unet_features,
                               temperature=args.init_temperature,
                               beta_size=args.beta_size,)
    theta_encoder = ThetaEncoder(theta_size=args.theta_size,
                                 img_size=datasets['train'].scan_shape,
                                 kernel_size=args.kernel_size)
    beta_discriminant = BetaDiscriminant(beta_size=args.beta_size,
                                        img_size=datasets['train'].scan_shape,
                                        kernel_size=args.kernel_size)
    decoder = Decoder(unet_init_features=args.unet_features,
                      beta_size=args.beta_size,
                      theta_size=args.theta_size,)
    model = BaseModel(beta_encoder, theta_encoder, decoder, beta_discriminant, args)
    if args.checkpoint is not None:
        model.load_from_checkpoint(args.checkpoint)
    # ------------
    # logger
    # ------------
    tb_logger = pl_loggers.TensorBoardLogger(logger_path)
    args.logger = tb_logger
    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    if not args.test_only:
        trainer.fit(model, loaders['train'], loaders['eval'])

        # ------------
        # testing
        # ------------
        for key in ['train', 'eval']:
            result = trainer.test(test_dataloaders=loaders[key])
            print(result)
    else:
        for key in loaders:
            result = trainer.test(model, test_dataloaders=loaders[key])
            print(result)


if __name__ == '__main__':
    main()
