import os
import codecs
from libs.constants import DEFAULT_ENCODING
import csv
from libs.utils import convertPointsToXY

TARGET_FILE = 'faceset.csv'
FIELD_NAMES = ['path', 'xmin', 'ymin', 'xmax', 'ymax', 'name', 'drawingFlag']


class OneFileWriter:
    def __init__(self, foldername, filename, imgSize, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.localImgPath = localImgPath
        self.verified = False
        self.shapes = None

        self.fieldnames = FIELD_NAMES

    def save(self, shapes, imagepath, targetFile=TARGET_FILE):
        self.deleteExistentImagepath(imagepath, targetFile)
        with open(targetFile, mode='a', newline='') as f:
            writer = csv.DictWriter(f, self.fieldnames)
            for shape in shapes:
                p = []
                p.append(round(shape.points[0].x()))
                p.append(round(shape.points[0].y()))
                p.append(round(shape.points[2].x()))
                p.append(round(shape.points[2].y()))
                writer.writerow(
                    {'path': imagepath, 'xmin': p[0], 'ymin': p[1], 'xmax': p[2], 'ymax': p[3], 'name': shape.label,
                     'drawingFlag': shape.drawingFlag})

    def deleteExistentImagepath(self, imagePath, targetFile):
        lines = []
        if not os.path.exists(targetFile):
            return
        with open(targetFile, mode='r') as r:
            reader = csv.DictReader(r, self.fieldnames)
            for row in reader:
                if row['path'] != imagePath:
                    lines.append(row)
        os.remove(targetFile)
        with open(targetFile, mode='a', newline='') as f:
            writer = csv.DictWriter(f, self.fieldnames)
            writer.writerows(lines)


class OneFileReader:
    def __init__(self, ):
        self.shapes = None

        self.fieldnames = FIELD_NAMES

    def loadShapes(self, imagePath, targetFile=TARGET_FILE):
        shapes = []
        if not os.path.exists(targetFile):
            return
        with open(targetFile, mode='r') as r:
            reader = csv.DictReader(r, self.fieldnames)
            for row in reader:
                if row['path'] == imagePath:
                    shapes.append([int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['name'],
                                  row['drawingFlag']])
        return shapes
