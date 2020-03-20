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


def cache(impath, model, cachepath):
    if os.path.exists(cachepath):
        return
    os.makedirs(os.path.split(cachepath)[0])



if __name__ == '__main__':
    model = Model(ResNet(predict=True))
    model.compile(torch.optim.SGD(model.model.parameters(),
                                  lr=0.001,
                                  momentum=0.9,
                                  weight_decay=1e-4),
                  ContrastiveLoss(),
                  metric=None,
                  device='cuda')
    model.load_weights('/home/palm/PycharmProjects/seven2/snapshots/pairs/1/epoch_0_0.05799773925956032.pth')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])

    target_path = '/home/palm/PycharmProjects/seven/images/cropped/unknown/obj'
    query_path = '/home/palm/PycharmProjects/seven/images/cropped/train'
    for target_image_path in os.listdir(target_path):
        target = os.path.join(target_path, target_image_path)
        target_image_ori = Image.open(target)
        target_image = transform(target_image_ori)
        minimum = (float('inf'), 0)
        for query_folder in os.listdir(query_path):
            for query_image_path in os.listdir(os.path.join(query_path, query_folder)):
                query = os.path.join(query_path, query_folder, query_image_path)
                query_image = Image.open(query)
                query_image = transform(query_image)
                x = torch.zeros((1, 2, 3, 224, 224))
                x[0, 0] = target_image
                x[0, 1] = query_image
                y = model.predict(x)
                if y < minimum[0]:
                    minimum = (y, query_folder)
        if minimum[0] < 1:
            os.makedirs(os.path.join('/home/palm/PycharmProjects/seven/images/cropped/unknown', minimum[1]), exist_ok=True)
            target_image_ori.save(os.path.join('/home/palm/PycharmProjects/seven/images/cropped/unknown', minimum[1], target_image_path))
