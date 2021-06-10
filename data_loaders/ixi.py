from torch.utils.data.dataset import Dataset
import torch
import pandas as pd
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import pickle



class IXILoader(Dataset):
    def __init__(self,
                 csv_path,
                 voxel_size=2,
                 n_slices=1,
                 modalities=('T1w', 'T2w'),
                 preload=False,
                 preloaded_path=None):
        self.df = pd.read_csv(csv_path)
        self.voxel_size = voxel_size
        self.n_slices = n_slices
        self.modalities = list(modalities)
        self._preload = preload
        if preloaded_path is not None:
            with open(preloaded_path, 'rb') as f:
                self.outputs = pickle.load(f)

        elif self._preload:
            self.preload()



    def __len__(self):
        return len(self.df)

    @property
    def scan_shape(self):
        batch = self.__getitem__(0)
        return tuple(batch['scans'].shape[-2:])

    def __getitem__(self, idx):
        if self._preload:
            return self.outputs[idx]
        else:
            return self.getitem(idx)

    def getitem(self, idx):
        output = {'scans': [],
                  'site': []}
        # repeat data extraction for each modality
        for i, mod in enumerate(self.modalities):
            path = self.df.loc[idx, mod]
            img = sitk.ReadImage(path)
            img = sitk.GetArrayFromImage(img)
            # resample image
            img = img[::self.voxel_size, ::self.voxel_size, ::self.voxel_size]
            # normalise pixel intensities
            img[img>0] = (img[img>0] - img[img>0].mean()) / img[img>0].std()
            # extract slices
            starting_slice = int((img.shape[0] - self.n_slices) / 2)
            imgs = np.stack([img[starting_slice + slice]
                             for slice in range(self.n_slices)])
            output['scans'].append(imgs[:, np.newaxis])
            site = np.stack([self.df.loc[idx, 'subDS'] == 'Guys'
                             for _ in range(self.n_slices)])
            output['site'].append(site)
        output = {key: np.stack(output[key], axis=1) for key in output}
        return {key: torch.Tensor(output[key]) for key in output}


    def preload(self):
        print('preloading ...')
        self.outputs = []
        for idx in tqdm(range(len(self.df))):
            self.outputs.append(self.getitem(idx))
        print('done preloading')



