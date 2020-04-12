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
from yolo.utils import create_csv_training_instances
import numpy as np
import cv2
import tensorflow as tf
import keras
from evaluate_util import evaluate, all_annotation_from_instance
import pickle as pk

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

    train_ints, valid_ints, labels, max_box_per_image = create_csv_training_instances(
        '/home/palm/PycharmProjects/seven2/anns/annotation.csv',
        '/home/palm/PycharmProjects/seven2/anns/val_ann.csv',
        '/home/palm/PycharmProjects/seven2/anns/classes.csv',
    )

    labels_to_names = [x.split(',')[0] for x in open('/home/palm/PycharmProjects/seven2/anns/classes.csv').read().split('\n')[:-1]]
    prediction_model = models.load_model('/home/palm/PycharmProjects/seven2/snapshots/infer_model_temp.h5')
    names_to_labels = {}
    for x in open('/home/palm/PycharmProjects/seven2/anns/classes.csv').read().split('\n')[:-1]:
        names_to_labels[x.split(',')[0]] = int(x.split(',')[1])
    query_path = '/home/palm/PycharmProjects/seven/images/cropped6/train'
    cache_path = '/home/palm/PycharmProjects/seven/caches'
    cache_dict = {}
    all_detections = []
    all_annotations = []

    for instance in valid_ints:
        all_annotation = all_annotation_from_instance(instance, names_to_labels)

        image = read_image_bgr(instance["filename"])

        # copy to draw ong
        draw = image.copy()

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=720, max_side=1280)

        # process image
        boxes, scores, labels = prediction_model.predict_on_batch(np.expand_dims(image, axis=0))

        # correct for image scale
        boxes /= scale
        all_detection = [[] for _ in labels_to_names]
        for box, score, _ in zip(boxes[0], scores[0], labels[0]):
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
            if minimum[0] < 1:
                label = names_to_labels[minimum[1]]
                all_detection[label].append([*b, score])
        # all_detection = np.array(all_detection, dtype='uint16')
        all_detections.append(all_detection)
        all_annotations.append(all_annotation)

    pk.dump([all_detections, all_annotations], open('siamese_cache.pk', 'wb'))
    average_precisions, total_instances = evaluate(all_detections, all_annotations, len(names_to_labels))
    print('mAP using the weighted average of precisions among classes: {:.4f}'.format(
        sum([a * b for a, b in zip(total_instances, average_precisions)]) / sum(total_instances)))
    for label, average_precision in average_precisions.items():
        print(['ov', 'mif'][label] + ': {:.4f}'.format(average_precision))
    print('mAP: {:.4f}'.format(sum(average_precisions) / sum(x for x in total_instances)))  # mAP: 0.5000
