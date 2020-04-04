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
import time
from lshash.lshash import LSHash


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


def memory_cache(cachedict, model, query, cachepath, transform):
    if cachepath not in cachedict:
        image = Image.open(query)
        image = transform(image)
        cachedict[cachepath] = load_cache(model, image, cachepath)
    return cachedict, cachedict[cachepath]


def memory_image(query, image_dict, transform):
    if query not in image_dict:
        query_image = Image.open(query)
        query_image = transform(query_image)
        image_dict[query] = query_image
    return image_dict, image_dict[query]


def predict_image(pil_image, model, query_path, cache_path, cache_dict):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])
    target_image = transform(pil_image)
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
    return minimum


if __name__ == '__main__':
    model = Model(ResNet(predict=True))
    model.compile(torch.optim.SGD(model.model.parameters(),
                                  lr=0.001,
                                  momentum=0.9,
                                  weight_decay=1e-4),
                  ContrastiveLoss(),
                  metric=None,
                  device='cuda')
    model.load_weights('/home/palm/PycharmProjects/seven2/snapshots/pairs/4/epoch_0_0.016697616640688282.pth')
    model.model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])

    lsh = LSHash(hash_size=16, input_dim=1024, num_hashtables=5)

    target_path = '/home/palm/PycharmProjects/seven/images/cropped2/unknown/obj'
    query_path = '/home/palm/PycharmProjects/seven/images/cropped2/train'
    cache_path = '/home/palm/PycharmProjects/seven/caches'
    cache_dict = {}
    image_dict = {}
    with torch.no_grad():
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
                    t = time.time()
                    query = os.path.join(query_path, query_folder, query_image_path)
                    image_dict, query_image = memory_image(query, image_dict, transform)
                    t1 = time.time() - t
                    cache_dict, query_features = memory_cache(cache_dict, model.model, query_image, os.path.join(cache_path, query_folder, query_image_path + '.pth'))
                    t2 = time.time() - t
                    y = lsh.euclidean_dist(target_features.cpu().numpy()[0], query_features.cpu().numpy()[0])
                    t3 = time.time() - t
                    print(t1, t2, t3)
                    if y < minimum[0]:
                        minimum = (y, query_folder)
            print(minimum, target_image_path)
            # if minimum[0] < 1.:
            #     os.makedirs(os.path.join('/home/palm/PycharmProjects/seven/images/cropped2/unknown', minimum[1]), exist_ok=True)
            #     target_image_ori.save(os.path.join('/home/palm/PycharmProjects/seven/images/cropped2/unknown', minimum[1], target_image_path))
