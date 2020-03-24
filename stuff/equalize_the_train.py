import shutil
import os


if __name__ == '__main__':
    src_dir = '/home/palm/PycharmProjects/seven/images/cropped/train'
    dst_dir = '/home/palm/PycharmProjects/seven/images/cropped2/train'
    for folder in os.listdir(src_dir):
        if folder == 'obj':
            continue
        for file in os.listdir(os.path.join(src_dir, folder)):
            if not os.path.exists(os.path.join(dst_dir, folder, file)):
                if not os.path.exists(os.path.join('/home/palm/PycharmProjects/seven/images/cropped2/unknown', 'obj', file)):
                    print(os.path.join('/home/palm/PycharmProjects/seven/images/cropped2/unknown', 'obj', file), 'is not exit')
                else:
                    os.makedirs(os.path.join(dst_dir, folder), exist_ok=True)
                    shutil.copy(os.path.join('/home/palm/PycharmProjects/seven/images/cropped2/unknown', 'obj', file),
                                os.path.join(dst_dir, folder, file))
