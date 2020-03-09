import os
from xml.etree import cElementTree as ET
from PIL import Image


if __name__ == '__main__':
    ann_folder = '/media/palm/data/7/ann1-30-9'
    # names = open('../names.txt').read().split('\n')[:-1]
    anns = open('/home/palm/PycharmProjects/Seven/stuff/data1-30-9.txt').read().split('\n')[:-1]
    # assert len(anns) == len(names)
    for i in range(len(anns)):
        x = anns[i].split(' ')
        imname = os.path.join(*x[0].split('/')[-2:])
        impath = x[0]
        try:
            image = Image.open(impath)
        except FileNotFoundError:
            continue
        width, height = image.size
        root = ET.Element('annotation')
        ET.SubElement(root, 'filename').text = imname
        size = ET.SubElement(root, 'size')
        ET.SubElement(size, 'width').text = str(width)
        ET.SubElement(size, 'height').text = str(height)

        obj_ = anns[i].split(' ')
        if len(obj_) > 5:
            continue
        ctxt = obj_[0].split('/')[-2]
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = ctxt

        x1 = min(480, max(0, min(int(obj_[1]), int(obj_[2]))))
        x2 = min(480, max(0, max(int(obj_[1]), int(obj_[2]))))
        y1 = min(640, max(0, min(int(obj_[3]), int(obj_[4]))))
        y2 = min(640, max(0, max(int(obj_[3]), int(obj_[4]))))

        bndbx = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbx, 'xmin').text = str(x1)
        ET.SubElement(bndbx, 'xmax').text = str(x2)
        ET.SubElement(bndbx, 'ymin').text = str(y1)
        ET.SubElement(bndbx, 'ymax').text = str(y2)
        tree = ET.ElementTree(root)
        if abs(x1-x2) < 10 or abs(y1-y2) < 10:
            continue
        tree.write(os.path.join(ann_folder, imname[:-4].replace('/', '_')+'.xml'))
