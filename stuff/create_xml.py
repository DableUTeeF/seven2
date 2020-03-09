import os
from xml.etree import cElementTree as ET
from PIL import Image
import json


if __name__ == '__main__':
    txt_folder = '/media/palm/data/7/txt/'
    ann_folder = '/media/palm/data/7/anns/'
    image_folder = '/media/palm/data/7/images/'
    names = open('/home/palm/PycharmProjects/Seven/stuff/obj.names').read().split('\n')
    for txt in os.listdir(txt_folder):
        imname = txt[:-4]+'.jpg'
        try:
            image = Image.open(os.path.join(image_folder, imname))
        except FileNotFoundError:
            continue
        width, height = image.size
        root = ET.Element('annotation')
        ET.SubElement(root, 'filename').text = imname
        ET.SubElement(root, 'path').text = os.path.join(image_folder, imname)
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)

        class_txt = open(os.path.join(txt_folder, txt)).read().split('\n')
        for obj_txt in class_txt:
            if len(obj_txt) == 0:
                break
            obj_ = obj_txt.split(' ')

            obj = ET.SubElement(root, 'object')
            ET.SubElement(obj, 'name').text = names[int(obj_[0])]

            w = int(float(obj_[3]) * width)
            h = int(float(obj_[4]) * height)
            x = int(float(obj_[1]) * width) - int(w / 2)
            y = int(float(obj_[2]) * height) - int(h / 2)

            bndbx = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbx, 'xmin').text = str(x)
            ET.SubElement(bndbx, 'xmax').text = str(x+w)
            ET.SubElement(bndbx, 'ymin').text = str(y)
            ET.SubElement(bndbx, 'ymax').text = str(y+h)
        tree = ET.ElementTree(root)
        tree.write(os.path.join(ann_folder, txt[:-4]+'.xml'))
