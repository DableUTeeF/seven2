"""
pairwise = 3.2582782537205974e-05, 3.403033881831663e-05
lsh_pair = 1.1194194062374617e-05, 1.2179314307509757e-05
lsh_cosd = 1.3693449689933570e-05, 1.3121787751072310e-05
lsh_eusq = 0.8139138601596410e-05, 0.8484695964420297e-05
lsh_eusc = 3.4231815388225626e-05, 3.3993883803595526e-05
lsh_hamm = 2.0442149504858307e-05, 2.0106958167212748e-05
new_pair = 1.6684917586866188e-05, 1.6523444134256113e-05
"""
from lshash.lshash import LSHash
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
import numpy as np


def euclidean_dist_new(x, y):
    """ This is a hot function, hence some optimizations are made. """
    result = np.dot(x, x) + np.dot(y, y) - np.dot(x, y) * 2
    return np.sqrt(result)

def euclidean_dist(x, y):
    """ This is a hot function, hence some optimizations are made. """
    diff = np.array(x) - y
    return np.sqrt(np.dot(diff, diff))


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

    lsh = LSHash(hash_size=16, input_dim=1024, num_hashtables=5)

    cache_folder = '/home/palm/PycharmProjects/seven/caches'
    with torch.no_grad():
        target_image_ori = Image.open('/home/palm/PycharmProjects/seven/images/cropped2/unknown/obj/0_036.jpg')
        target_image = transform(target_image_ori)
        x = torch.zeros((1, 3, 224, 224))
        x[0] = target_image
        target_features = model.model._forward_impl(x.cuda()).cpu()
        minimum = (float('inf'), 0)
        ts = []
        # for class_folder in os.listdir(cache_folder):
        #     for file in os.listdir(os.path.join(cache_folder, class_folder)):
        #         cache = torch.load(os.path.join(cache_folder, class_folder, file)).cpu()
        #         lsh.index(cache[0])
        # target_hash = lsh._hash(lsh.uniform_planes[0], target_features[0])
        for class_folder in os.listdir(cache_folder):
            for file in os.listdir(os.path.join(cache_folder, class_folder)):
                t = time.time()
                cache = torch.load(os.path.join(cache_folder, class_folder, file)).cpu()
                t1 = time.time() - t
                # query_hash = lsh._hash(lsh.uniform_planes[0], cache[0])
                t2 = time.time() - t
                # distant = lsh.hamming_dist(target_hash, query_hash)
                distant = euclidean_dist_new(target_features.numpy()[0], cache.numpy()[0])
                # distant = torch.pairwise_distance(cache, target_features)
                t3 = time.time() - t
                # print(t1, t2, t3)
                ts.append(t3-t2)
                if distant < minimum[0]:
                    minimum = (distant, class_folder)
        print(minimum)
        print(sum(ts) / len(ts))
        a = 0
