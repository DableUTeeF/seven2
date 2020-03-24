import os
import sys

# noinspection PyUnboundLocalVariable
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    __package__ = "siamese"
from siamese.models import ResNet, ContrastiveLoss
from PIL import Image
import torch
from natthaphon import Model
from torchvision import transforms
from xml.etree import cElementTree as ET


def save_cache(model, image, cachepath):
    os.makedirs(os.path.split(cachepath)[0], exist_ok=True)
    x = torch.zeros((1, 3, 224, 224))
    x[0] = image
    out = model._forward_impl(x.cuda())
    torch.save(out, cachepath)
    return out


def load_cache(model, image, cachepath):
    if os.path.exists(cachepath):
        return torch.load(cachepath)
    return save_cache(model, image, cachepath)


def memory_cache(cachedict, model, image, cachepath):
    if cachepath not in cachedict:
        cachedict[cachepath] = load_cache(model, image, cachepath)
    return cachedict, cachedict[cachepath]


if __name__ == '__main__':
    model = Model(ResNet(predict=True))
    model.compile(torch.optim.SGD(model.model.parameters(),
                                  lr=0.001,
                                  momentum=0.9,
                                  weight_decay=1e-4),
                  ContrastiveLoss(),
                  metric=None,
                  device='cuda')
    model.load_weights('/home/palm/PycharmProjects/seven2/snapshots/pairs/3/epoch_0_0.03454810580774366.pth')
    model.model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])


