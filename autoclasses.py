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
from xml.etree import cElementTree as ET
import numpy as np
import cv2
import tensorflow as tf
import keras
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.set_session(sess)

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

    labels_to_names = {0: 'obj'}
    prediction_model = models.load_model('/home/palm/PycharmProjects/seven2/snapshots/infer_model_temp.h5')

    target_path = '/home/palm/PycharmProjects/seven/images/cropped2/unknown/obj'
    query_path = '/home/palm/PycharmProjects/seven/images/cropped2/train'
    cache_path = '/home/palm/PycharmProjects/seven/caches'
    cache_dict = {}
    image_dict = {}
    for set_name in [0, 1, 2, 3]:
        folder = f'/home/palm/PycharmProjects/seven/data1/{set_name}'
        anns_path = f'/home/palm/PycharmProjects/seven2/xmls/revised/{set_name}'
        exiting_anns = [os.path.basename(x) for x in os.listdir(anns_path)]
        for i in os.listdir(folder):
            if i[:-4] + '.xml' in exiting_anns:
                continue
            if 'txt' in i:
                continue
            image = read_image_bgr(os.path.join(folder, i))

            # copy to draw ong
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            # preprocess image for network
            image = preprocess_image(image)
            image, scale = resize_image(image, min_side=720, max_side=1280)

            # process image
            boxes, scores, labels = prediction_model.predict_on_batch(np.expand_dims(image, axis=0))

            # correct for image scale
            boxes /= scale

            root = ET.Element('annotation')
            ET.SubElement(root, 'filename').text = i
            ET.SubElement(root, 'path').text = os.path.join(folder, i)
            size = ET.SubElement(root, 'size')
            ET.SubElement(size, 'width').text = str(draw.shape[1])
            ET.SubElement(size, 'height').text = str(draw.shape[0])

            for box, score, label in zip(boxes[0], scores[0], labels[0]):
                # scores are sorted so we can break
                if score < 0.5:
                    continue
                b = box.astype(int)
                minimum = (float('inf'), 0)
                with torch.no_grad():
                    target_image_ori = Image.fromarray(draw[b[1]:b[3], b[0]:b[2]])
                    target_image = transform(target_image_ori)
                    x = torch.zeros((1, 3, 224, 224))
                    x[0] = target_image
                    target_features = model.model._forward_impl(x.cuda())
                    for query_folder in os.listdir(query_path):
                        for query_image_path in os.listdir(os.path.join(query_path, query_folder)):
                            query = os.path.join(query_path, query_folder, query_image_path)
                            image_dict, query_image = memory_image(query, image_dict, transform)
                            cache_dict, query_features = memory_cache(cache_dict, model.model, query_image, os.path.join(cache_path, query_folder, query_image_path + '.pth'))
                            y = LSHash.euclidean_dist(target_features.cpu().numpy()[0], query_features.cpu().numpy()[0])
                            if y < minimum[0]:
                                minimum = (y, query_folder)
                if minimum[0] > 1:
                    minimum = (minimum[0], 'obj')
                print(minimum)
                obj = ET.SubElement(root, 'object')
                ET.SubElement(obj, 'name').text = minimum[1]
                bndbx = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbx, 'xmin').text = str(b[0])
                ET.SubElement(bndbx, 'ymin').text = str(b[1])
                ET.SubElement(bndbx, 'xmax').text = str(b[2])
                ET.SubElement(bndbx, 'ymax').text = str(b[3])

            # cv2.imshow(f'im_{i}', draw)
            tree = ET.ElementTree(root)
            os.makedirs(f'/home/palm/PycharmProjects/seven2/xmls/classed/{set_name}/', exist_ok=True)
            tree.write(f'/home/palm/PycharmProjects/seven2/xmls/classed/{set_name}/' + i[:-4] + '.xml')

