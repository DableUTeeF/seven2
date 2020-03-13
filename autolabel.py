from retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from retinanet.utils.visualization import draw_box, draw_caption
from retinanet.utils.colors import label_color
from retinanet import models
import cv2
import os
import numpy as np
import time
from xml.etree import cElementTree as ET


if __name__ == '__main__':
    labels_to_names = {0: 'obj'}
    set_name = 0
    prediction_model = models.load_model('/home/palm/PycharmProjects/seven2/snapshots/infer_model_temp.h5')

    folder = f'/home/palm/PycharmProjects/seven/data1/{set_name}'
    anns_path = f'/home/palm/PycharmProjects/seven/anns/{set_name}'
    exiting_anns = [os.path.basename(x) for x in os.listdir(anns_path)]
    for i in os.listdir(folder):
        if i[:-4] + '.xml' in exiting_anns:
            continue
        if 'txt' in i:
            continue
        image = read_image_bgr(os.path.join(folder, i))

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=720, max_side=1280)

        # process image
        start = time.time()
        boxes, scores, labels = prediction_model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

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
                print('nope', score)
                break
            b = box.astype(int)

            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = labels_to_names[label]
            bndbx = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbx, 'xmin').text = str(b[0])
            ET.SubElement(bndbx, 'ymin').text = str(b[1])
            ET.SubElement(bndbx, 'xmax').text = str(b[2])
            ET.SubElement(bndbx, 'ymax').text = str(b[3])

            color = label_color(label)
            draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
        # cv2.imshow(f'im_{i}', draw)
        tree = ET.ElementTree(root)
        tree.write(f'/home/palm/PycharmProjects/seven2/anns/{set_name}/' + i[:-4] + '.xml')

