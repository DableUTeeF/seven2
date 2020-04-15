import os
from xml.etree import cElementTree as ET
import tensorflow as tf
import keras

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.set_session(sess)

if __name__ == '__main__':

    for set_name in [0, 1, 2, 3]:
        folder = f'/home/palm/PycharmProjects/seven/data1/'
        anns_path = f'/home/palm/PycharmProjects/seven2/xmls/readjusted/{set_name}'
        dst = '/home/palm/PycharmProjects/seven2/xmls/final'
        for file in os.listdir(anns_path):
            tree = ET.parse(os.path.join(anns_path, file))
            for elem in tree.iter():
                if 'filename' in elem.tag:
                    elem.text = f'{set_name}_{elem.text}'
                if 'path' in elem.tag:
                    elem.text = f'/home/palm/PycharmProjects/seven/data1/{set_name}_'+os.path.basename(elem.text)
                if 'name' in elem.tag:
                    if elem.text == 'Almond_bar':
                        elem.text = 'United Almond 19g'
                    elif elem.text == 'Diva 160ml':
                        elem.text = 'Daiwa dishwashing liquid lemon 160ml'
                    elif elem.text == 'Protractor ruler':
                        elem.text = 'TD protractor'
                    elif elem.text == 'Soffell Flora 80ml':
                        elem.text = 'Soffel Flora 80ml'
                    elif elem.text == 'Soffel flora 8ml':
                        elem.text = 'Soffel Lotion flora 8ml'
                    elif elem.text == 'Kitkat thai tea':
                        elem.text = 'Kitkat red 35g'
                    elif elem.text == 'KitKat Milktea 35g':
                        elem.text = 'Kitkat red 35g'
                    elif elem.text == 'KitKat Red 35g':
                        elem.text = 'Kitkat red 35g'
                    elif elem.text == 'Koh-kae salted peanuts 42g':
                        elem.text = 'Koh-Kae Salted Peanuts 42g'
                    elif elem.text == 'Almind_fried_56g':
                        elem.text = 'Almond_fried_56g'
                    elif 'Darlie' in elem.text:
                        elem.text = 'Darlie green'


            tree.write(f'/home/palm/PycharmProjects/seven2/xmls/final/{set_name}_' + file[:-4] + '.xml')

