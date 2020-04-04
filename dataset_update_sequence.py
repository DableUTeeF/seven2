import os


os.system('python xml2csv.py')
os.system('python siamese/create_dataset.py')
os.system('python stuff/equalize_the_train.py')
# os.system('python autoclasses.py')
