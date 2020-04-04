import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
from siamese.siamese_predict import memory_cache
import os
import multiprocessing
from functools import partial
from contextlib import contextmanager


@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()

# nope
def predict_image_class(query_folder, target_features, cache_dict, class_minimum):
    minimum = (float('inf'), 0)
    for query_image_path in os.listdir(os.path.join(query_path, query_folder)):
        t = time.time()
        query = os.path.join(query_path, query_folder, query_image_path)
        t1 = time.time() - t
        cache_dict, query_features = memory_cache(cache_dict, model.model, query, os.path.join(cache_path, query_folder, query_image_path + '.pth'), transform)
        t2 = time.time() - t
        y = LSHash.euclidean_dist(target_features.cpu().numpy()[0], query_features.cpu().numpy()[0])
        t3 = time.time() - t
        print(t1, t2, t3)
        if y < minimum[0]:
            minimum = (y, query_folder)
    class_minimum[query_folder] = minimum


if __name__ == '__main__':
    model = Model(ResNet(predict=True))
    model.compile(torch.optim.SGD(model.model.parameters(),
                                  lr=0.001,
                                  momentum=0.9,
                                  weight_decay=1e-4),
                  ContrastiveLoss(),
                  metric=None,
                  device='cpu')
    model.load_weights('/home/palm/PycharmProjects/seven2/snapshots/pairs/4/epoch_0_0.016697616640688282.pth')
    model.model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])
    query_path = '/home/palm/PycharmProjects/seven/images/cropped2/train'
    cache_path = '/home/palm/PycharmProjects/seven/caches'
    target_path = '/home/palm/PycharmProjects/seven/images/cropped2/unknown/obj'
    cache_dict = {}
    with torch.no_grad():
        for target_image_path in os.listdir(target_path):
            target = os.path.join(target_path, target_image_path)
            target_image_ori = Image.open(target)
            target_image = transform(target_image_ori)
            x = torch.zeros((1, 3, 224, 224))
            x[0] = target_image
            target_features = model.model._forward_impl(x)
            minimum = (float('inf'), 0)
            query_folders = os.listdir(query_path)
            class_minimum = {}
            with poolcontext(processes=8) as pool:
                results = pool.map(partial(predict_image_class, target_features=target_features, cache_dict=cache_dict, class_minimum=class_minimum), query_folders)
            print(class_minimum)
