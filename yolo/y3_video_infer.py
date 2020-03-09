import cv2
import os
import sys
# noinspection PyUnboundLocalVariable
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    __package__ = "yolo"
from yolo.y3frontend import *
import json
import time
from yolo.utils import draw_boxesv3, normalize, evaluate, evaluate_coco, get_yolo_boxes, parse_annotation, create_csv_training_instances
from PIL import Image
import numpy as np
from yolo.preprocessing import minmaxresize, Y3BatchGenerator
from keras.models import load_model

if __name__ == '__main__':

    config_path = '/home/palm/PycharmProjects/seven2/yolo/sevenconfig.json'

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())
    train_ints, valid_ints, labels, max_box_per_image = create_csv_training_instances(
        config['train']['train_csv'],
        config['valid']['valid_csv'],
        config['train']['classes_csv'],
    )

    infer_model = yolo3(
            fe='effnetb3',
            output_type='dw',
            nb_class=len(labels)
    )

    infer_model.load_weights('/home/palm/PycharmProjects/seven2/snapshots/22_4.1761_1.2766.h5',
                             # by_name=True,
                             # skip_mismatch=True,
                             )

    path = "/media/palm/data/coco/images/val2017"
    pad = 1
    # for _ in range(1000):
    # if 1:

    valid_generator = Y3BatchGenerator(
        instances=valid_ints,
        anchors=config['model']['anchors'],
        labels=labels,
        downsample=32,
        max_box_per_image=max_box_per_image,
        batch_size=1,
        min_net_size=config['model']['min_input_size'],
        max_net_size=config['model']['max_input_size'],
        shuffle=True,
        jitter=0.0,
    )
    cap = cv2.VideoCapture(1)
    t = time.time()
    while 1:

        _, image = cap.read()
        x = time.time()
        # filename = '001dxxyile2uxkblr99uqo6fuhgprpccznlze0z0djhs9gkek2tsm8u5hsfzx62o.jpg'
        # filename = 'download.jpeg'
        image, w, h = minmaxresize(image, 416, 608)
        # image = cv2.resize(image, (416, 416))
        if pad:
            imsize = image.shape
            if imsize[0] > imsize[1]:
                tempim = np.zeros((imsize[0], imsize[0], 3), dtype='uint8')
                distant = (imsize[0] - imsize[1]) // 2
                tempim[:, distant:distant + imsize[1], :] = image
                image = tempim
                h = imsize[0]
                w = imsize[0]

            elif imsize[1] > imsize[0]:
                tempim = np.zeros((imsize[1], imsize[1], 3), dtype='uint8')
                distant = (imsize[1] - imsize[0]) // 2
                tempim[distant:distant + imsize[0], :, :] = image
                image = tempim
                h = imsize[1]
                w = imsize[1]

        image = np.expand_dims(image, 0)

        boxes = get_yolo_boxes(infer_model,
                               image,
                               608, 608,  # todo: change here too
                               config['model']['anchors'],
                               0.5,
                               0.5)[0]
        # infer_model.predict(image)
        # labels = ['badhelmet', 'badshoes', 'goodhelmet', 'goodshoes', 'person']
        # # draw bounding boxes on the image using labels
        image = draw_boxesv3(image[0], boxes, labels, 0.75)
        cv2.imshow('img', image.astype('uint8'))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

