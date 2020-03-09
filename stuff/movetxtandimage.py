import shutil
import os


if __name__ == '__main__':
    image_dest = '/media/palm/data/7/images'
    anns_dest = '/media/palm/data/7/anns'
    root_folder = '/media/palm/data/7/data/'
    for folder in os.listdir(root_folder):
        # if not os.path.isdir(os.path.join(anns_dest, folder)):
        #     os.mkdir(os.path.join(anns_dest, folder))
        # if not os.path.isdir(os.path.join(image_dest, folder)):
        #     os.mkdir(os.path.join(image_dest, folder))
        for file in os.listdir(os.path.join(root_folder, folder)):
            if file[-4:] == '.txt':
                shutil.move(os.path.join(root_folder, folder, file),
                            os.path.join(anns_dest, file))
            else:
                shutil.move(os.path.join(root_folder, folder, file),
                            os.path.join(image_dest, file))
