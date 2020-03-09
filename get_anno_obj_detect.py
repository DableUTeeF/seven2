import cv2
import numpy as np
import os

N = 3
path = './data1/'

cap = [None] * N
for i in range(N):
    cap[i] = cv2.VideoCapture(i + 1)
    cap[i].set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap[i].set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
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

class_name = '0'
class_num = 0
while True:
    img = [None] * N
    drawimg = [None] * N
    for i in range(N):
        _, img[i] = cap[i].read()
        img[i] = np.rot90(img[i])
        drawimg[i] = img[i].copy()
        cv2.rectangle(drawimg[i], (ix[i], iy[i]), (ex[i], ey[i]), (0, 0, 255), 2)
        cv2.putText(drawimg[i], class_name, (ix[i], iy[i] - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        cv2.imshow('img' + str(i), cv2.resize(drawimg[i], None, None, 0.2, 0.2))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('n'):
        class_num += 1
        class_name = str(class_num)
    if key == ord('s'):
        if not os.path.exists(path + class_name):
            os.makedirs(path + class_name)
        maxnum = 0
        for f in os.listdir(path + class_name):
            if '.jpg' in f:
                n = int(f[:-4])
                if n > maxnum:
                    maxnum = n
        for i in range(N):
            fn = path + class_name + '/' + str(maxnum + 1 + i) + '.jpg'
            cv2.imwrite(fn, img[i])
            fn = path + class_name + '/' + str(maxnum + 1 + i) + '.txt'
            with open(fn, 'w') as f:
                f.write('%s %d %d %d %d\n' % (class_name, ix[i], iy[i], ex[i], ey[i]))
