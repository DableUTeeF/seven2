from yolo.utils import create_csv_training_instances
import cv2
import os

if __name__ == '__main__':
    train_ints, valid_ints, labels, max_box_per_image = create_csv_training_instances(
        '/home/palm/PycharmProjects/seven2/anns/ann.csv',
        '/home/palm/PycharmProjects/seven2/anns/val_ann.csv',
        '/home/palm/PycharmProjects/seven2/anns/c.csv',
    )
    save_path = '/home/palm/PycharmProjects/seven/images/cropped'
    for instance in train_ints:
        image = cv2.imread(instance['filename'])
        for idx, obj in enumerate(instance['object']):
            x1 = max(0, obj['xmin'])
            x2 = min(image.shape[1], obj['xmax'])
            y1 = max(0, obj['ymin'])
            y2 = min(image.shape[0], obj['ymax'])
            center = (x1 + (x2-x1)//2, y1 + (y2-y1)//2)
            p = max(x2-x1, y2-y1) // 2
            x1 = max(0, center[0] - p)
            x2 = min(image.shape[1], center[0] + p)
            y1 = max(0, center[1] - p)
            y2 = min(image.shape[0], center[1] + p)
            cropped_image = image[y1:y2, x1:x2]
            if '134.jpg' in instance['filename']:
                print()
            os.makedirs(os.path.join(save_path, 'train', obj['name']), exist_ok=True)
            cv2.imwrite(os.path.join(save_path, 'train', obj['name'], str(idx) + os.path.basename(instance['filename'])),
                        cropped_image)
