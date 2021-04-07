import os
import codecs
from libs.constants import DEFAULT_ENCODING
import csv
from libs.utils import convertPointsToXY

TARGET_FILE = 'faceset.csv'
FIELD_NAMES = ['path', 'xmin', 'ymin', 'xmax', 'ymax', 'id']


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

    def save(self, shapes, pathid, targetFile=TARGET_FILE, subject_dictionary=None):
        self.deleteExistentImagepath(pathid, targetFile)
        with open(targetFile, mode='a', newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, self.fieldnames)
            i = 0
            for shape in shapes:
                p = []
                p.append(round(shape.points[0].x()))
                p.append(round(shape.points[0].y()))
                p.append(round(shape.points[2].x()))
                p.append(round(shape.points[2].y()))
                id = subject_dictionary[shape.label]
                writer.writerow(
                    {'path': pathid, 'xmin': p[0], 'ymin': p[1], 'xmax': p[2], 'ymax': p[3], 'id': id, })
                i += 1

    def deleteExistentImagepath(self, pathid, targetFile):
        lines = []
        if not os.path.exists(targetFile):
            return
        with open(targetFile, mode='r', encoding='utf-8') as r:
            reader = csv.DictReader(r, self.fieldnames)
            for row in reader:
                if int(row['path']) != pathid:
                    lines.append(row)
        os.remove(targetFile)
        with open(targetFile, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, self.fieldnames)
            writer.writerows(lines)


class OneFileReader:
    def __init__(self, ):
        self.shapes = None

        self.fieldnames = FIELD_NAMES

    def loadShapes(self, imagePath, targetFile=TARGET_FILE, subjects_dictionary=None, path_to_id=None):
        shapes = []
        if not os.path.exists(targetFile):
            return
        if 'labelImg-master' in imagePath:
            imagePath = imagePath.split('labelImg-master')[-1][1:]
        pathid = int(path_to_id[imagePath])
        with open(targetFile, mode='r', encoding='utf-8') as r:
            reader = csv.DictReader(r, self.fieldnames)
            for row in reader:

                if int(row['path']) == pathid:
                    name = subjects_dictionary[int(row['id'])]
                    shapes.append([int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), name, ])
        return shapes
