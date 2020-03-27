import shutil
import os


if __name__ == '__main__':
    src_dir = '/home/palm/PycharmProjects/seven/images/cropped2/train'
    dst_dir = '/home/palm/PycharmProjects/seven/images/cropped3/train'
    for folder in os.listdir(src_dir):
        if folder not in os.listdir(dst_dir):
            shutil.copytree(os.path.join(src_dir, folder),
                            os.path.join('/home/palm/PycharmProjects/seven/images/cropped3/test', folder))
        else:
            print(folder)
