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

if __name__ == '__main__':
    model = Model(ResNet(predict=True))
    model.compile(torch.optim.SGD(model.model.parameters(),
                                  lr=0.001,
                                  momentum=0.9,
                                  weight_decay=1e-4),
                  ContrastiveLoss(),
                  metric=None,
                  device='cuda')
    model.load_weights('/home/palm/PycharmProjects/algea/snapshots/pairs/2/epoch_5_1.2236453857421874.pth')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((112, 112)),
                                    transforms.ToTensor(),
                                    normalize])

    target = '/media/palm/data/MicroAlgae/16_8_62/cropped/val/mif/0MIF eggs-kato-40x (121).jpg'
    query = '/media/palm/data/MicroAlgae/16_8_62/cropped/train/ov/0OV egg kato 40X (794).jpg'
    target_image = Image.open(target)
    query_image = Image.open(query)
    target_image = transform(target_image)
    query_image = transform(query_image)
    x = torch.zeros((1, 2, 3, 112, 112))
    x[0, 0] = target_image
    x[0, 1] = query_image
    y = model.predict(x)
    print(y)
