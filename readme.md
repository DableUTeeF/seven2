# Installation
Note: The tested environment is Ubuntu 18.04 and Python 3.6
* Install RetinaNet dependencies by `pip install -r requirement.txt`
* Install Siamese Network dependencies by `pip install torch torchvision` and `pip install dependencies/natthaphon-1.0.0-py3-none-any.whl`

# How to train
* Note: the commands presented here are using seven2, the root directory of this repo,  as working directory
* Step 1: Prepare xml
  * Copy all the xml to somewhere you want in one place, default `xmls`
  * Edit the paths in `xml2csv_classify.py` and `xml2csv_detect.py` to match your directory structure. The default structure looks like this.
```
     |-- xmls                             |-- images
     |-- |-- 0                            |-- |-- 0
     |-- |-- |-- 01.xml                   |-- |-- |-- 01.jpg
     |-- |-- |-- 02.xml                   |-- |-- |-- 02.jpg
     |-- |-- 1                            |-- |-- 1
     |-- |-- |-- 01.xml                   |-- |-- |-- 01.jpg
     |-- |-- |-- 02.xml                   |-- |-- |-- 02.jpg
```
* Step 2: Create necessary dataset from xml
  * To train the detector, run `xml2csv_detect.py` this will create 2 csv files, `anns/ann.csv` and `anns/c.csv`
  * To train the Siamese Network classifier, run `dataset_update_sequence.py`. This will create 3 csv files, `anns/annotation.csv`, `anns/classes.csv` and `anns/val_ann.csv` and then create the small images of each object in `images/cropped/train`
* Step 3: Train detector
  * To train RetinaNet detector, run `python retinanet/bin/train.py --snapshot-path snapshots/temp --epochs 1 csv anns/ann.csv anns/c.csv `. You can change `--epochs 1` to however many epochs you want to train.
  * Convert train model to inference model by`retinanet/bin/convert_model.py snapshots/temp/resnet50_csv_01.h5  snapshots/detector.h5`. If training epochs is more than 1, you can change the `resnet50_csv_01.h5` to other files in `snapshots/temp` as well.
  * The weights file to use later is `snapshots/detector.h5`
* Step 4: Train recognizer
   * Download the [`base.pth`](https://drive.google.com/open?id=1k64kFbubZl_xZoX0XgK2qIzaw4C71qJh) and copy it to `snapshots`
   * Run `python siamese/siamese_train.py`
   * The final weights is `snapshots/recognizer.pth` but other weights in `snapshots/pairs` are usable too.