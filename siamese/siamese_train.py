from siamese.models import ResNet, ContrastiveLoss
from siamese.datagen import SiameseCifarLoader
import os
import json
import sys
import torch
from torch.nn import functional as F
from natthaphon import Model
# noinspection PyUnboundLocalVariable
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    __package__ = "siamese"


class ThresholdAcc:
    def __call__(self, inputs, targets):
        distant = F.cosine_similarity(inputs[0], inputs[1])
        predict = (distant > 0.7).long()
        acc = torch.sum(predict != targets.long()).float() / targets.size(0)
        return acc

    def __str__(self):
        return 'acc()'


if __name__ == '__main__':
    model = Model(ResNet())
    model.compile(torch.optim.SGD(model.model.parameters(),
                                  lr=0.01,
                                  momentum=0.9,
                                  weight_decay=1e-4),
                  ContrastiveLoss(),
                  metric=ThresholdAcc(),
                  device='cuda')
    datagen = SiameseCifarLoader(os.path.join(rootpath, name))
    train_generator = datagen.get_trainset(64, 1)
    val_geerator = datagen.get_testset(100, 1)
    h = model.fit_generator(train_generator, 200, validation_data=val_geerator, schedule=[100, 150], tensorboard=f'logs/pair/{len(os.listdir("logs/pair"))}')
    model.save_weights('/home/palm/PycharmProjects/seven2/snapshots/pair/1/temp.pth')
    with open('siamese.json', 'w') as wr:
        json.dump(h, wr)
