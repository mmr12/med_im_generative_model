from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader
import sys, os
import torch
import pickle
sys.path.append(os.getcwd())
from baselines.DeweyEtAl.LightningModule import BaseModel
from baselines.DeweyEtAl.model import Encoder, Decoder
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
    parser.add_argument('--small_unet', default=False, type=bool)
    parser.add_argument('--out_channels', default=10, type=int)
    parser.add_argument('--unet_features', default=4, type=int)
    parser.add_argument('--beta_size', default=5, type=int)
    parser.add_argument('--theta_size', default=2, type=int)
    parser.add_argument('--init_temperature', default=1, type=float)
    parser.add_argument('--decrease_every', default=5, type=int)
    parser.add_argument('--decrease_by', default=.1, type=float)
    parser.add_argument('--decrease_min', default=.5, type=float)
    # other stuff
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--test_only', default=False, type=bool)
    parser.add_argument('--save_preload', default=True, type=bool)

    # ------------
    # collect args
    # ------------
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BaseModel.add_model_specific_args(parser)
    args = parser.parse_args()
    # housekeeping
    torch.autograd.set_detect_anomaly(True)
    pl.seed_everything(args.seed)
    logger_path = 'lightning_logs/Dewey/{}/'.format(args.data)
    # GPU
    if args.gpus is not None and isinstance(args.gpus, int):
        # Make sure that it only uses a single GPU..
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
        args.gpus = 1

    # --------------
    # Load data
    # --------------
    data_files = {key: args.csv_path + key + '.csv' for key in ['train', 'eval', 'test']}
    preloaded_paths = {key: '/vol/biomedic2/mmr12/GitLab_projects/harmonisation/data/{}_{}voxel_{}slice_{}.pkl' \
        .format(args.data, args.voxel_size,args.n_slices, key) for key in ['train', 'eval', 'test']}
    for key in data_files:
        if not os.path.isfile(data_files[key]):
            raise KeyError
        if not os.path.isfile(preloaded_paths[key]):
            preloaded_paths[key] = None
        else:
            args.save_preload = False
    shuffle = {'train': True, 'eval': False, 'test': False}
    datasets = {key: IXILoader(data_files[key], args.voxel_size, args.n_slices, args.modalities, args.preload,
                               preloaded_paths[key]) for key in data_files}
    loaders = {key: DataLoader(datasets[key], batch_size=args.batch_size, num_workers=8, shuffle=shuffle[key])
               for key in datasets}

    # save preloaded files
    if args.save_preload:
        preloaded_paths = {key: '/vol/biomedic2/mmr12/GitLab_projects/harmonisation/data/{}_{}voxel_{}slice_{}.pkl' \
            .format(args.data, args.voxel_size, args.n_slices, key) for key in ['train', 'eval', 'test']}
        for key in datasets:
            with open(preloaded_paths[key], 'rb') as f:
                pickle.dump(datasets[key].outputs, f)

    # -------------
    # Load model
    # -------------
    encoder = Encoder(unet_out_channels=args.out_channels,
                      unet_init_features=args.unet_features,
                      beta_size=args.beta_size,
                      temperature=args.init_temperature,
                      theta_size=args.theta_size,
                      img_size=datasets['train'].scan_shape,
                      small_unet=args.small_unet)
    decoder = Decoder(unet_init_features=args.unet_features,
                      beta_size=args.beta_size,
                      theta_size=args.theta_size,
                      small_unet=args.small_unet)
    model = BaseModel(encoder, decoder, args)
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
