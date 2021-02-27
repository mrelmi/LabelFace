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

    def showWindow(self, similarity_numbers, similarity_labels, similarity_urls):
        for i in range(similarity_numbers):
            self.newFace(i, similarity_labels[i], similarity_urls[i])
        self.newInput(similarity_numbers)
        self.w.setWindowTitle("choose")
        self.w.show()
        self.w.exec_()

    def newFace(self, i, name, url):
        image = cv2.imread(url)
        image = cv2.resize(image, (self.image_size, self.image_size))
        cv2.imwrite(str(i) + '.jpg', image)
        url = str(i) + '.jpg'

        self.buttons.append(QPushButton(self.w))
        self.buttons[i].setStyleSheet(BUTTON_CSS.replace('xxxxx.jpg', url))
        self.buttons[i].setGeometry(i * self.image_size + self.image_pad, self.image_pad, self.image_size, self.image_size)
        self.buttons[i].clicked.connect(partial(self.p, name))

        self.labels.append(QLabel(parent=self.w, text=name))
        self.labels[i].setGeometry(i * self.image_size + 2 * self.image_pad, self.image_size + self.image_pad // 2, 40, 17)

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
        self.buttons[i].setGeometry(i * self.image_size + 3 * self.image_pad // 2, self.image_size / 2 + self.image_pad // 2,
                                    self.image_size - 6*self.image_pad, 2 * self.image_pad)
        self.buttons[i].clicked.connect(self.getNewOne)
