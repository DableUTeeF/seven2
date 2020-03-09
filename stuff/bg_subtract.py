import cv2
import numpy as np
import os

if __name__ == '__main__':
    path = '/media/palm/data/7/data1-30-9-gs/data1/300'
    gt = [cv2.imread(os.path.join(path, f'{x}.jpg')) for x in [1, 2, 3]]
    for imid in range(12):
        image = [cv2.imread(os.path.join(path, f'{x}.jpg')) for x in [1+(1+imid)*3, 2+(1+imid)*3, 3+(1+imid)*3]]
        for idx in range(3):
            diff = np.abs(cv2.cvtColor(gt[idx], cv2.COLOR_BGR2GRAY) - cv2.cvtColor(image[idx], cv2.COLOR_BGR2GRAY))
            mask = cv2.threshold(diff, 128, 255, cv2.THRESH_BINARY)[1]
            mask = cv2.dilate(mask, None, iterations=1)
            mask[:350] = 255
            # mask[:, :70] = 255
            # mask[:, 400:] = 255
            contours, hierarchy = cv2.findContours(mask.astype('uint8'), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

            contours_poly = [None] * len(contours)
            boundRect = [None] * len(contours)
            pt = [float('inf'), float('inf'), -1, -1]  # x, y, width, height
            for i, c in enumerate(contours):
                contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                boundRect[i] = cv2.boundingRect(contours_poly[i])

                if boundRect[i][0] == 0 or boundRect[i][2] >= 400:
                    continue
                if pt[0] > boundRect[i][0]:
                    pt[0] = boundRect[i][0]
                if pt[1] > boundRect[i][1]:
                    pt[1] = boundRect[i][1]
                if pt[2] < boundRect[i][2] + boundRect[i][0]:
                    pt[2] = boundRect[i][2] + boundRect[i][0]
                if pt[3] < boundRect[i][3] + boundRect[i][1]:
                    pt[3] = boundRect[i][3] + boundRect[i][1]

            # cv2.rectangle(image, (pt[0], pt[1]), (pt[2], pt[3]), (230, 180, 128))
            cv2.rectangle(image[idx], (pt[0], pt[1]), (pt[2], pt[3]), (128, 128, 255), 2)
            cv2.imwrite(f'/home/palm/PycharmProjects/Seven/out/1/{idx+1+(1+imid)*3}.jpg', image[idx])


    # cv2.imshow(f'gt{idx}', gt[idx])
    # cv2.imshow(f'mask{idx}', mask)
    # while 1:
    #     keyboard = cv2.waitKey()
    #     print(keyboard)
    #     if keyboard == 113:
    #         break
