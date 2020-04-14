import shutil
import os


if __name__ == '__main__':
    src_dir = '/home/palm/PycharmProjects/seven/images/test6/train'
    dst_root = '/home/palm/PycharmProjects/seven/images/cropped6'
    dst_dir = os.path.join(dst_root, 'train')
    for folder in os.listdir(src_dir):
        if folder not in os.listdir(dst_dir):
            shutil.copytree(os.path.join(src_dir, folder),
                            os.path.join(dst_root, 'test', folder))
        else:
            print(folder)
