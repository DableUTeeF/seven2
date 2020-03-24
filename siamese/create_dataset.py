import os
import sys

# noinspection PyUnboundLocalVariable
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    __package__ = "siamese"
from yolo.utils import create_csv_training_instances
import cv2
import os

if __name__ == '__main__':
    train_ints, valid_ints, labels, max_box_per_image = create_csv_training_instances('/home/palm/PycharmProjects/seven2/anns/ann.csv',
                                                                                      '/home/palm/PycharmProjects/seven2/anns/val_ann.csv',
                                                                                      '/home/palm/PycharmProjects/seven2/anns/c.csv',
                                                                                      )
    save_path = '/home/palm/PycharmProjects/seven/images/cropped2'
    for instance in train_ints:
        image = cv2.imread(instance['filename'])
        for idx, obj in enumerate(instance['object']):
            x1 = max(0, obj['xmin'])
            x2 = min(image.shape[1], obj['xmax'])
            y1 = max(0, obj['ymin'])
            y2 = min(image.shape[0], obj['ymax'])

            cropped_image = image[y1:y2, x1:x2]
            if x2 - x1 > y2 - y1:
                p = ((x2 - x1) - (y2 - y1)) // 2
                cropped_image = cv2.copyMakeBorder(cropped_image, p, p, 0, 0, cv2.BORDER_CONSTANT)
            else:
                p = ((y2 - y1) - (x2 - x1)) // 2
                cropped_image = cv2.copyMakeBorder(cropped_image, 0, 0, p, p, cv2.BORDER_CONSTANT)

            setname = os.path.split(instance['filename'])[0][-1]
            if obj['name'] == 'obj':
                os.makedirs(os.path.join('/home/palm/PycharmProjects/seven/images/cropped2/unknown/obj'), exist_ok=True)
                cv2.imwrite(os.path.join('/home/palm/PycharmProjects/seven/images/cropped2/unknown/obj', setname + '_' + str(idx) + os.path.basename(instance['filename'])),
                            cropped_image)
            else:
                os.makedirs(os.path.join(save_path, 'train', obj['name']), exist_ok=True)
                cv2.imwrite(os.path.join(save_path, 'train', obj['name'], setname + '_' + str(idx) + os.path.basename(instance['filename'])),
                            cropped_image)
