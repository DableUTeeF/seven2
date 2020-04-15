import cv2
from six import raise_from
import csv
import sys



def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """ Parse the classes file given by csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _read_annotations(csv_reader, classes):
    """ Read annotations from the csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, x1, y1, x2, y2, class_name = row[:6]
        except ValueError:
            raise_from(ValueError(
                'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)),
                       None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
    return result


def _open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


def create_csv_training_instances(train_csv, test_csv, class_csv, with_wh=False):
    with _open_for_csv(class_csv) as file:
        classes = _read_classes(csv.reader(file, delimiter=','))
    with _open_for_csv(train_csv) as file:
        train_image_data = _read_annotations(csv.reader(file, delimiter=','), classes)
    with _open_for_csv(test_csv) as file:
        test_image_data = _read_annotations(csv.reader(file, delimiter=','), classes)
    train_ints = []
    valid_ints = []
    labels = list(classes)
    max_box_per_image = 0
    for k in train_image_data:
        image_data = train_image_data[k]
        ints = {'filename': k, 'object': []}
        if with_wh:
            x = cv2.imread(k)
            height, width, _ = x.shape
            ints['width'] = width
            ints['height'] = height
        for i, obj in enumerate(image_data):
            o = {'xmin': obj['x1'], 'xmax': obj['x2'], 'ymin': obj['y1'], 'ymax': obj['y2'], 'name': obj['class']}
            ints['object'].append(o)
            if i + 1 > max_box_per_image:
                max_box_per_image = i + 1
        train_ints.append(ints)

    for k in test_image_data:
        image_data = test_image_data[k]
        ints = {'filename': k, 'object': []}
        if with_wh:
            x = cv2.imread(k)
            height, width, _ = x.shape
            ints['width'] = width
            ints['height'] = height
        for i, obj in enumerate(image_data):
            o = {'xmin': obj['x1'], 'xmax': obj['x2'], 'ymin': obj['y1'], 'ymax': obj['y2'], 'name': obj['class']}
            ints['object'].append(o)
            if i + 1 > max_box_per_image:
                max_box_per_image = i + 1
        valid_ints.append(ints)

    return train_ints, valid_ints, sorted(labels), max_box_per_image
