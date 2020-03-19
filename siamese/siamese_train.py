import os
import sys

# noinspection PyUnboundLocalVariable
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    __package__ = "siamese"
from siamese.models import ResNet, ContrastiveLoss
from siamese.datagen import DirectorySiameseLoader
import json
import torch
from torch.nn import functional as F
from natthaphon import Model
from torchvision import transforms


class ThresholdAcc:
    def __call__(self, inputs, targets):
        distant = F.cosine_similarity(inputs[0], inputs[1])
        predict = (distant > 0.7).long()
        acc = torch.sum(predict != targets.long()).float() / targets.size(0)
        return acc

    def __str__(self):
        return 'acc()'


if __name__ == '__main__':
    save_no = len(os.listdir('/home/palm/PycharmProjects/seven2/snapshots/pairs'))

    model = Model(ResNet(zero_init_residual=False))
    model.compile(torch.optim.SGD(model.model.parameters(),
                                  lr=0.001,
                                  momentum=0.9,
                                  weight_decay=1e-4),
                  ContrastiveLoss(),
                  metric=None,
                  device='cuda')
    model.load_weights('/home/palm/PycharmProjects/seven2/snapshots/base.pth', load_opt=False)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_datagen = DirectorySiameseLoader('/home/palm/PycharmProjects/seven/images/cropped/train',
                                           transforms.Compose([transforms.Resize(256),
                                                               transforms.RandomResizedCrop(224),
                                                               transforms.RandomHorizontalFlip(),
                                                               transforms.RandomVerticalFlip(),
                                                               transforms.ToTensor(),
                                                               normalize]))
    train_generator = train_datagen.get_dset(8, 1)
    os.makedirs(f'/home/palm/PycharmProjects/seven2/snapshots/pairs/{save_no}', exist_ok=True)
    h = model.fit_generator(train_generator, 20,
                            schedule=[10, 15],
                            tensorboard=f'logs/pair/{len(os.listdir("logs/pair"))}',
                            epoch_end=model.checkpoint(f'/home/palm/PycharmProjects/seven2/snapshots/pairs/{save_no}', 'val_ContrastiveLoss'))
    model.save_weights('/home/palm/PycharmProjects/seven2/snapshots/pairs_temp.pth')
    with open('siamese.json', 'w') as wr:
        json.dump(h, wr)
