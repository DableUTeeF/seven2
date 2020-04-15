from xml.etree import cElementTree as ET
import os


if __name__ == '__main__':
    open('anns/val_ann.csv', 'w')
    open('anns/classes.csv', 'w')
    classes = []
    trainset = []
    testset = []
    images_base_path = 'images/'
    with open('anns/annotation.csv', 'w') as wr:
        for set_name in [0, 1, 2, 3]:
            path = f'./xmls/readjusted/{set_name}'
            for file in os.listdir(path):
                val = False
                if set_name == 1:
                    val = True
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
                            impath = os.path.join(images_base_path, str(set_name), basename)
                    if 'object' in elem.tag:
                        if cls != '' and (xmax+xmin+ymax+ymax) != 0 and impath != 0:
                            if cls not in classes:
                                with open('anns/classes.csv', 'a') as cwr:
                                    cwr.write(f'{cls},{len(classes)}\n')
                                classes.append(cls)
                            ln = f'{impath},{xmin},{ymin},{xmax},{ymax},{cls}'
                            if val:
                                testset.append(impath)
                                with open('anns/val_ann.csv', 'a') as vwr:
                                    vwr.write(ln)
                                    vwr.write('\n')
                            else:
                                trainset.append(impath)
                                wr.write(ln)
                                wr.write('\n')
                    elif 'name' in elem.tag:
                        cls = elem.text
                        if cls == 'Almond_bar':
                            cls = 'United Almond 19g'
                        elif cls == 'Diva 160ml':
                            cls = 'Daiwa dishwashing liquid lemon 160ml'
                        elif cls == 'Protractor ruler':
                            cls = 'TD protractor'
                        elif cls == 'Soffell Flora 80ml':
                            cls = 'Soffel Flora 80ml'
                        elif cls == 'Soffel flora 8ml':
                            cls = 'Soffel Lotion flora 8ml'
                        elif cls == 'Kitkat thai tea':
                            cls = 'Kitkat red 35g'
                        elif cls == 'KitKat Milktea 35g':
                            cls = 'Kitkat red 35g'
                        elif cls == 'KitKat Red 35g':
                            cls = 'Kitkat red 35g'
                        elif cls == 'Koh-kae salted peanuts 42g':
                            cls = 'Koh-Kae Salted Peanuts 42g'
                        elif cls == 'Almind_fried_56g':
                            cls = 'Almond_fried_56g'
                        elif 'Darlie' in cls:
                            cls = 'Darlie green'
                    elif 'xmin' in elem.tag:
                        xmin = elem.text
                    elif 'ymin' in elem.tag:
                        ymin = elem.text
                    elif 'xmax' in elem.tag:
                        xmax = elem.text
                    elif 'ymax' in elem.tag:
                        ymax = elem.text
                if 1: # cls != 'obj':
                    if cls not in classes:
                        with open('anns/classes.csv', 'a') as cwr:
                            cwr.write(f'{cls},{len(classes)}\n')
                        classes.append(cls)
                    ln = f'{impath},{xmin},{ymin},{xmax},{ymax},{cls}'
                    if val:
                        testset.append(impath)
                        with open('anns/val_ann.csv', 'a') as vwr:
                            vwr.write(ln)
                            vwr.write('\n')
                    else:
                        trainset.append(impath)
                        wr.write(ln)
                        wr.write('\n')
    print(len(set(trainset)))
    print(len(set(testset)))
