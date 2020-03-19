from torch.utils.data import DataLoader
import numpy as np
from torchvision.datasets import ImageFolder
import torch

class DirectorySiameseLoader:
    def __init__(self, target_path, transform):
        self.dset = self.DataSet(target_path, transform)

    def get_dset(self, batch_size, num_worker, shuffle=True):
        return self.Loader(self.dset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_worker)

    class DataSet:
        def __init__(self, target_path, transform):
            self.target_path = target_path

            self.dset = ImageFolder(target_path,
                                    transform=transform)
            self.len = len(self.dset)
            self.curidx = -1
            self.setidx = -1

        def __next__(self):
            self.curidx += 1
            self.setidx += 1
            if self.setidx >= self.len:
                self.setidx -= self.len
            return self[self.curidx]

        def __len__(self):
            return self.len**2

        def __getitem__(self, idx):
            # query image
            xq, y_1 = self.dset[idx % self.len]
            x = torch.zeros((2, *xq.size()))
            x[0] = xq
            # target image
            xt, y_2 = self.dset[idx // self.len]
            x[1] = xt
            y = y_1 != y_2
            return x, y

    class Loader(DataLoader):
        def __len__(self):
            return int(np.round(len(self.dataset) / self.batch_size))


if __name__ == '__main__':
    train_datagen = DirectorySiameseLoader('/media/palm/data/MicroAlgae/16_8_62/cropped/train',
                                           None)
    train_generator = train_datagen.get_dset(16, 1)
    s = train_datagen.dset
    s1 = s[1]
