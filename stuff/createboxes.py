import cv2
import numpy as np
import os

N = 3
drawing = False
ix = [None] * N
iy = [None] * N
ex = [None] * N
ey = [None] * N
for i in range(N):
    ix[i], iy[i] = 0, 0
    ex[i], ey[i] = 0, 0


def draw_rect(event, x, y, flags, i):
    global ix, iy, ex, ey, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix[i], iy[i] = x, y
        ex[i], ey[i] = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            drawimg[i] = img[i].copy()
            ex[i] = x
            ey[i] = y
            # cv2.rectangle(drawimg[i], (ix[i], iy[i]), (x, y), (255, 255, 255), 1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex[i] = x
        ey[i] = y
        # cv2.rectangle(drawimg[i], (ix[i], iy[i]), (x, y), (255, 255, 255), 1)
        if ix[i] < x and iy[i] < y:
            ix[i], ex[i], iy[i], ey[i] = ix[i], x, iy[i], y
        else:
            ex[i], ix[i], ey[i], iy[i] = ix[i], x, iy[i], y


for i in range(N):
    cv2.namedWindow('img' + str(i))
    cv2.setMouseCallback('img' + str(i), draw_rect, i)
class_name = ''
root_directory = 'E:\Work\Garbage' #name of the folder
source_dir = 'data1' #
img = []
name = []
try:
    written = [x.split(' ')[0] for x in open('data1-30-9.txt', 'r').readlines()]
except FileNotFoundError:
    written = []
for subdir in os.listdir(os.path.join(root_directory, source_dir)):
    s = sorted(os.listdir(os.path.join(root_directory, source_dir, subdir)))
    subdir_list = {}
    for a in s:
        subdir_list[f'{int(a.split(".")[0]):02d}.{a.split(".")[1]}'] = a
    slist = sorted(subdir_list)
    for fs in slist:
        files = subdir_list[fs]
        if 'txt' in files:
            continue
        if os.path.join(root_directory, source_dir, subdir, files) in written:
            continue
        img.append(cv2.imread(os.path.join(root_directory, source_dir, subdir, files)))
        name.append(os.path.join(root_directory, source_dir, subdir, files))
        while len(img) == N:
            drawimg = [None] * N
            for i in range(N):
                drawimg[i] = img[i].copy()
                cv2.rectangle(drawimg[i], (ix[i], iy[i]), (ex[i], ey[i]), (0, 0, 255), 2)
                cv2.putText(drawimg[i], class_name, (ix[i], iy[i] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                cv2.imshow('img' + str(i), drawimg[i])
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                raise SystemExit
            if key == ord('s'):
                img = []
                with open('data1-30-9-gs.txt', 'a') as f:
                    for i in range(N):
                        f.write('%s %d %d %d %d\n' % (name[i],
                                                      ix[i], iy[i], ex[i], ey[i]))
                name = []
                break
