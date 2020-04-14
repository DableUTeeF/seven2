import cv2
import numpy as np
import tkinter as tk
import threading
from PIL import Image
from PIL import ImageTk
import os
from retinanet import models
import time
from retinanet.utils.image import preprocess_image, resize_image
import tensorflow as tf
from univ_utils import add_bbox
from threading import Thread
import cv2


class WebcamThread:

    def __init__(self, src=0, name="WebcamThread", af=None, f=None, w=None, h=None):
        self.cap = cv2.VideoCapture(src)
        if af is not None:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        if f is not None:
            self.cap.set(cv2.CAP_PROP_FOCUS, 0)
        if f is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        if f is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        _, self.frame = self.cap.read()
        self.name = name
        self.stopped = False

    def update(self):
        while True:
            if self.stopped:
                return
            _, self.frame = self.cap.read()

    def start(self):
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


def verify(facevector1, facevector2, threshold=0.4):
    min_dist = 99999
    dist = np.linalg.norm(np.subtract(facevector1, facevector2))

    if dist < min_dist:
        min_dist = dist
    print(min_dist)

    if min_dist > threshold:
        return False
    else:
        return True


class APP:
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.thread = None
        self.stopEvent = None

        self.model = models.load_model('/home/palm/PycharmProjects/seven2/snapshots/infer_model_temp.h5')
        self.graph = tf.get_default_graph()
        self.classes = {}

        self.root = tk.Tk()
        self.root.configure(background='SlateGray4')
        self.root.bind('<KeyRelease>', self.keydetect)
        self.panel = None
        self.qrscanner = ''
        self.predictionLabel = tk.Text(self.root, height=30, width=40,
                                       borderwidth=0, highlightthickness=0,
                                       relief='ridge', background="SlateGray4", foreground='SlateGray1')
        self.predictionLabel.grid(row=0, column=0, padx=4, pady=2)
        self.classLabel = tk.Text(self.root, height=30, width=40,
                                  borderwidth=0, highlightthickness=0,
                                  relief='ridge', background="SlateGray4", foreground='SlateGray1')
        self.classLabel.grid(row=0, column=2, padx=4, pady=2)

        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.vdoLoop, args=())
        self.thread.start()

        self.root.wm_title("BingoBox")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        self.t = time.time()
        self.weight = 210

    def vdoLoop(self):
        with self.graph.as_default():
            while not self.stopEvent.is_set():
                obj = {}
                frame = cv2.imread(f'/home/palm/PycharmProjects/seven/data1/1/1.jpg')

                draw = frame.copy()
                image = preprocess_image(frame)
                image, scale = resize_image(image, min_side=720, max_side=1280)
                boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))

                boxes /= scale

                for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    if score < 0.9:
                        break

                    b = box.astype(int)
                    draw = add_bbox(draw, b, label, self.labels_to_names, score)
                    if label not in obj:
                        obj[label] = 0
                    obj[label] += 1

                blk = ImageTk.PhotoImage(Image.fromarray(cv2.resize(draw, (360, 640))[..., ::-1]))
                if self.weight > 10:
                    color = 'chartreuse3' if abs(self.get_weight(obj) - self.weight) < 20 else 'orangered'
                else:
                    color = 'cornflower blue'
                if self.panel is None:
                    self.panel = tk.Label(image=blk, borderwidth=0, highlightthickness=3, highlightbackground=color)
                    self.panel.image = blk
                    self.panel.grid(row=0, column=1, padx=2, pady=2)
                else:
                    self.panel.configure(image=blk, highlightthickness=3, relief="solid", highlightbackground=color)
                    self.panel.image = blk

                self.predictionLabel.config(state='normal')
                self.predictionLabel.delete(1.0, tk.END)
                self.predictionLabel.insert(tk.END, f"Obj{' ' * 7}Qty{' ' * 7}Ttl wt. \n")

    def get_weight(self, obj):
        weights = 0
        for o in obj:
            weights += self.labels_to_weight[o] * obj[o]
        return weights

    def keydetect(self, e):
        if e.char == 'q':
            self.onClose()

    def onClose(self):
        print("close")
        self.stopEvent.set()
        self.cap.stop()
        self.root.quit()
        os.system('killall python')


if __name__ == '__main__':
    cap = WebcamThread(0, "QR detector 1", 0, 0, 1920, 1080).start()
    app = APP(cap)
    app.root.mainloop()
