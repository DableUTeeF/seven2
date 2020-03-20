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
    model.load_weights('/home/palm/PycharmProjects/seven2/snapshots/pairs/1/epoch_0_0.05799773925956032.pth')
    model.model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])

    target_path = '/home/palm/PycharmProjects/seven/images/cropped/unknown/obj'
    query_path = '/home/palm/PycharmProjects/seven/images/cropped/train'
    cache_path = '/home/palm/PycharmProjects/seven/caches'
    cache_dict = {}
    for target_image_path in os.listdir(target_path):
        target = os.path.join(target_path, target_image_path)
        target_image_ori = Image.open(target)
        target_image = transform(target_image_ori)
        x = torch.zeros((1, 3, 224, 224))
        x[0] = target_image
        target_features = model.model._forward_impl(x.cuda())
        minimum = (float('inf'), 0)
        for query_folder in os.listdir(query_path):
            for query_image_path in os.listdir(os.path.join(query_path, query_folder)):
                query = os.path.join(query_path, query_folder, query_image_path)
                query_image = Image.open(query)
                query_image = transform(query_image)
                cache_dict, query_features = memory_cache(cache_dict, model.model, query_image, os.path.join(cache_path, query_folder, query_image_path + '.pth'))
                y = torch.pairwise_distance(query_features.cpu(), target_features.cpu()).detach().numpy()
                if y < minimum[0]:
                    minimum = (y, query_folder)
        print(minimum, target_image_path)
        if minimum[0] < 1:
            os.makedirs(os.path.join('/home/palm/PycharmProjects/seven/images/cropped/unknown', minimum[1]), exist_ok=True)
            target_image_ori.save(os.path.join('/home/palm/PycharmProjects/seven/images/cropped/unknown', minimum[1], target_image_path))
