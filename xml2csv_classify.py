from xml.etree import cElementTree as ET
import os


if __name__ == '__main__':
    open('anns/classes.csv', 'w')
    classes = []
    trainset = []
    testset = []
    images_base_path = 'images/'
    with open('anns/annotation.csv', 'w') as wr:
        path = './xmls/readjusted/'
        for file in os.listdir(path):
            tree = ET.parse(os.path.join(path, file))
            if len(tree.findall('object')) == 0:
                continue
            ln = ''
            cls = ''
            xmin = 0
            xmax = 0
            ymin = 0
            ymax = 0
            impath = ''
            for elem in tree.iter():
                if 'path' in elem.tag:
                    impath = elem.text
                    if 'palm' not in impath:
                        if '\\' in impath:
                            basename = impath.split('\\')[-1]
                        else:
                            basename = os.path.basename(impath)
                        impath = os.path.join(images_base_path, basename)
                if 'object' in elem.tag:
                    if cls != '' and (xmax+xmin+ymax+ymax) != 0 and impath != 0:
                        if cls not in classes:
                            with open('anns/classes.csv', 'a') as cwr:
                                cwr.write(f'{cls},{len(classes)}\n')
                            classes.append(cls)
                        ln = f'{impath},{xmin},{ymin},{xmax},{ymax},{cls}'
                        trainset.append(impath)
                        wr.write(ln)
                        wr.write('\n')
                elif 'name' in elem.tag:
                    cls = elem.text
                elif 'xmin' in elem.tag:
                    xmin = elem.text
                elif 'ymin' in elem.tag:
                    ymin = elem.text
                elif 'xmax' in elem.tag:
                    xmax = elem.text
                elif 'ymax' in elem.tag:
                    ymax = elem.text
            if cls not in classes:
                with open('anns/classes.csv', 'a') as cwr:
                    cwr.write(f'{cls},{len(classes)}\n')
                classes.append(cls)
            ln = f'{impath},{xmin},{ymin},{xmax},{ymax},{cls}'
            trainset.append(impath)
            wr.write(ln)
            wr.write('\n')
    print(len(set(trainset)))
    print(len(set(testset)))
