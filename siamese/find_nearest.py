import os
import sys

# noinspection PyUnboundLocalVariable
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    __package__ = "siamese"
from siamese.models import ResNet, ContrastiveLoss
from siamese.siamese_predict import memory_image, memory_cache
from PIL import Image
from retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from lshash.lshash import LSHash
from retinanet import models
import torch
from natthaphon import Model
from torchvision import transforms
from boxutils import add_bbox
import numpy as np
import cv2
import tensorflow as tf
import keras
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.set_session(sess)


def pad(cropped_image, b):
    x1, y1 , x2, y2 = b
    if x2 - x1 > y2 - y1:
        p = ((x2 - x1) - (y2 - y1)) // 2
        cropped_image = cv2.copyMakeBorder(cropped_image, p, p, 0, 0, cv2.BORDER_CONSTANT)
    else:
        p = ((y2 - y1) - (x2 - x1)) // 2
        cropped_image = cv2.copyMakeBorder(cropped_image, 0, 0, p, p, cv2.BORDER_CONSTANT)
    return cropped_image

if __name__ == '__main__':
    model = Model(ResNet(predict=True))
    model.compile(torch.optim.SGD(model.model.parameters(),
                                  lr=0.001,
                                  momentum=0.9,
                                  weight_decay=1e-4),
                  ContrastiveLoss(),
                  metric=None,
                  device='cuda')
    model.load_weights('/home/palm/PycharmProjects/seven2/snapshots/pairs/5/epoch_1_0.012463876953125.pth')
    model.model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    normalize])

    labels_to_names = [x.split(',')[0] for x in open('/home/palm/PycharmProjects/seven2/anns/c.csv').read().split('\n')[:-1]]
    prediction_model = models.load_model('/home/palm/PycharmProjects/seven2/snapshots/infer_model_temp.h5')
    names_to_labels = {}
    nearest = {}
    for x in open('/home/palm/PycharmProjects/seven2/anns/c.csv').read().split('\n')[:-1]:
        names_to_labels[x.split(',')[0]] = int(x.split(',')[1])
        nearest[x.split(',')[0]] = []
    query_path = '/home/palm/PycharmProjects/seven/images/cropped6/train'
    cache_path = '/home/palm/PycharmProjects/seven/caches'
    cache_dict = {}
    dst = f'/home/palm/PycharmProjects/seven/predict/3'
    for set_name in [1]:
        folder = f'/home/palm/PycharmProjects/seven/data1/{set_name}'
        for i in os.listdir(folder):
            image = read_image_bgr(os.path.join(folder, i))

            # copy to draw ong
            draw = image.copy()

            # preprocess image for network
            image = preprocess_image(image)
            image, scale = resize_image(image, min_side=720, max_side=1280)

            # process image
            boxes, scores, labels = prediction_model.predict_on_batch(np.expand_dims(image, axis=0))

            # correct for image scale
            boxes /= scale

            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.5:
                    break
                b = box.astype(int)
                minimum = (float('inf'), 0)
                with torch.no_grad():
                    target_image_ori = pad(draw[b[1]:b[3], b[0]:b[2]], b)
                    target_image_ori = Image.fromarray(target_image_ori[..., ::-1])
                    target_image = transform(target_image_ori)
                    x = torch.zeros((1, 3, 224, 224))
                    x[0] = target_image
                    target_features = model.model._forward_impl(x.cuda())
                    for query_folder in os.listdir(query_path):
                        for query_image_path in os.listdir(os.path.join(query_path, query_folder)):
                            query = os.path.join(query_path, query_folder, query_image_path)
                            cache_dict, query_features = memory_cache(cache_dict, model.model, query, os.path.join(cache_path, query_folder, query_image_path + '.pth'), transform)
                            y = LSHash.euclidean_dist(target_features.cpu().numpy()[0], query_features.cpu().numpy()[0])
                            if y < minimum[0]:
                                minimum = (y, query_folder)
                if minimum[0] > 1:
                    minimum = (minimum[0], 'obj')
                label = names_to_labels[minimum[1]]
                draw = add_bbox(draw, b, label, labels_to_names, score)

            os.makedirs(dst, exist_ok=True)
            cv2.imwrite(os.path.join(dst, i), draw)
