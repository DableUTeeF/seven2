import os

files = ['stuff/data1(damkoeng).txt', 'stuff/data1-30-9.txt', 'stuff/data1_green_Screen.txt', 'stuff/data1-30-9-gs.txt']
dests = ['/home/root1/dataset-2020/7/data1/data1(damkoeng)',
         '/home/root1/dataset-2020/7/data1 (3)',
         '/home/root1/dataset-2020/7/data1/data1_green_Screen',
         '/home/root1/dataset-2020/7/data1 (2)',

         ]
classes = []
open('anns/val_ann.csv', 'w')
open('anns/ann.csv', 'w')
open('anns/classes.csv', 'w')
for i, file in enumerate(files):
    src = open(file).read().split('\n')
    while src[-1] == '':
        src = src[:-1]
    for line in src:
        ln = line.split(' ')
        s_paths = os.path.split(ln[0])
        cls = s_paths[0].split('/')[-1]
        d_path = os.path.join(dests[i], cls, s_paths[-1])
        if not os.path.exists(d_path):
            print(d_path, 'not exits')
            continue
        x1, y1, x2, y2 = int(ln[-4]), int(ln[-3]), int(ln[-2]), int(ln[-1])
        if (x2 - x1) + (y2 - y1) < 10:
            continue
        if x2 <= x1:
            x2 += 1
        if y2 <= y1:
            y2 += 1
        obj = f'{d_path},{min(x1, x2)},{min(y1, y2)},{max(x1, x2)},{max(y1, y2)},{s_paths[0]}'
        if s_paths[0] not in classes:
            with open('anns/val_ann.csv', 'a') as wr:
                wr.write(obj)
                wr.write('\n')
        else:
            with open('anns/ann.csv', 'a') as wr:
                wr.write(obj)
                wr.write('\n')

        classes.append(s_paths[0])
    classes = list(set(classes))
with open('anns/classes.csv', 'a') as wr:
    for i, line in enumerate(classes):
        wr.write(line)
        wr.write(',')
        wr.write(str(i))
        wr.write('\n')
