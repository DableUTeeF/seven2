from xml.etree import cElementTree as ET
import os


bad_img = [os.path.basename(x).split('_')[0] for x in open('/home/palm/PycharmProjects/tops/anns/bad_img.txt').read().split('\n')[:-1]]
def check_bad(file):
    return False
    x = os.path.basename(file)[:-4]
    return x in bad_img

if __name__ == '__main__':
    open('anns/val_ann.csv', 'w')
    open('anns/c.csv', 'w')
    classes = []
    with open('anns/ann.csv', 'w') as wr:
        for set_name in [0, 1]:
            folder = f'/home/palm/PycharmProjects/seven/data1/{set_name}'
            path = f'/home/palm/PycharmProjects/seven2/xmls/revised/{set_name}'
            for file in os.listdir(path):
                val = False
                if check_bad(file):
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
                        if 'nattakarnp' in impath:
                            impath.replace('/Users/nattakarnp/Desktop/seven/', '/home/root1/dataset-2020/seven/')
                    if 'object' in elem.tag:
                        if cls != '' and cls != 'obj' and (xmax+xmin+ymax+ymax) != 0 and impath != 0:
                            if cls not in classes:
                                with open('anns/c.csv', 'a') as cwr:
                                    cwr.write(f'{cls},{len(classes)}\n')
                                classes.append(cls)
                            ln = f'{impath},{xmin},{ymin},{xmax},{ymax},{cls}'
                            if val:
                                with open('anns/val_ann.csv', 'a') as vwr:
                                    vwr.write(ln)
                                    vwr.write('\n')
                            else:
                                wr.write(ln)
                                wr.write('\n')
                    elif 'name' in elem.tag:
                        cls = elem.text
                        if cls == 'Almond_bar':
                            cls = 'United Almond 19g'
                    elif 'xmin' in elem.tag:
                        xmin = elem.text
                    elif 'ymin' in elem.tag:
                        ymin = elem.text
                    elif 'xmax' in elem.tag:
                        xmax = elem.text
                    elif 'ymax' in elem.tag:
                        ymax = elem.text
                if cls != 'obj':
                    if cls not in classes:
                        with open('anns/c.csv', 'a') as cwr:
                            cwr.write(f'{cls},{len(classes)}\n')
                        classes.append(cls)
                    ln = f'{impath},{xmin},{ymin},{xmax},{ymax},{cls}'
                    if val:
                        with open('anns/val_ann.csv', 'a') as vwr:
                            vwr.write(ln)
                            vwr.write('\n')
                    else:
                        wr.write(ln)
                        wr.write('\n')
