import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras

from retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from retinanet.utils.visualization import draw_box, draw_caption
from retinanet.utils.colors import label_color
from retinanet import models
import cv2
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


if __name__ == '__main__':
    model_path = '/home/palm/PycharmProjects/seven2/snapshots/infer_model_temp.h5'
    labels_to_names = {0: 'obj'}
    model = models.load_model(model_path)

    # load image
    cap = cv2.VideoCapture(1)
    cap.set(28, 5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        _, image = cap.read()
        image = np.rot90(image)

        # copy to draw on
        draw = image.copy()
        # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=800, max_side=1333)

        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        boxes /= scale

        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break

            color = label_color(label)

            b = box.astype(int)
            draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)

        cv2.imshow('test', cv2.resize(draw, None, None, 0.3, 0.3))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

