from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from functools import partial
import sys

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

BUTTON_CSS = "background-image : url(xxxxx.jpg);margin: 1px; padding: 10px;\
                         background-color: \
                               rgba(255,255,0,255); \
                               color: rgba(0,0,0,255); \
                               border-style: solid; \
                               border-radius: 4px; border-width: 3px; \
                               border-color: rgba(0,0,0,255);"

IMAGE_SIZE = 200
IMAGE_PAD = 10


class PictureDialog:
    def __init__(self):
        self.w = QDialog()
        self.buttons = []
        self.labels = []
        self.clickedItem = None
        self.image_size = IMAGE_SIZE
        self.image_pad = IMAGE_PAD
        self.newId = None

    def showWindow(self, indexes, path):
        length = min(8, len(indexes))
        files = [f for f in listdir(path) if isfile(join(path, f, '0001.jpg'))]
        recomms = []

        indexes.sort(key=lambda tup: tup[0], reverse=True)
        for i in range(length):
            pics = [p for p in listdir(join(path, files[indexes[i][1]]))]
            recomms.append(
                (join(path, files[indexes[i][1]], pics[indexes[i][2]]), files[indexes[i][1]], indexes[i][0],
                 indexes[i][3]))

        for i in range(length):
            self.newFace(i, recomms)
        self.newInput(length)
        self.w.setWindowTitle("choose")
        self.w.show()
        self.w.exec_()

    def newFace(self, i, recomms, ):
        box = np.round(recomms[i][3][0][0]).astype(np.int16)
        image = cv2.imread(recomms[i][0])[box[1]:box[3], box[0]:box[2]]
        image = cv2.resize(image, (self.image_size, self.image_size))
        cv2.imwrite('temp/' + str(i) + '.jpg', image)
        url = 'temp/' + str(i) + '.jpg'

        self.buttons.append(QPushButton(self.w))
        self.buttons[i].setStyleSheet(BUTTON_CSS.replace('xxxxx.jpg', url))
        self.buttons[i].setGeometry(i * self.image_size + self.image_pad, self.image_pad, self.image_size,
                                    self.image_size)
        self.buttons[i].clicked.connect(partial(self.p, recomms[i][1]))

        self.labels.append(
            QLabel(parent=self.w, text='name :' + recomms[i][1] + '\nsimilarity : ' + str(round(recomms[i][2], 3))))

        self.labels[i].setGeometry(i * self.image_size + 3 * self.image_pad, self.image_size + self.image_pad,
                                   self.image_size,
                                   10 * self.image_pad)

    def p(self, name):
        self.clickedItem = name
        self.w.close()

    def getNewOne(self):
        name = QInputDialog(self.w)
        self.newId = name.getText(self.w, 'title', 'Enter new Name :')

    def newInput(self, i):
        self.buttons.append(QPushButton(self.w))
        self.buttons[i].setText('new')
        self.buttons[i].setStyleSheet("background-color: \
                               rgba(255,255,0,255); \
                               color: rgba(0,0,0,255); \
                               border-style: solid; \
                               border-radius: 7px; border-width: 5px; \
                               border-color: rgba(0,0,0,255);")
        self.buttons[i].setGeometry(i * self.image_size + 3 * self.image_pad // 2,
                                    self.image_size / 2 + self.image_pad // 2,
                                    self.image_size - 6 * self.image_pad, 2 * self.image_pad)
        self.buttons[i].clicked.connect(self.getNewOne)
