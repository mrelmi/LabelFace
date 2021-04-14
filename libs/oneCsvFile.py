import os
import codecs
from libs.constants import DEFAULT_ENCODING
import csv
from libs.utils import convertPointsToXY

TARGET_FILE = 'localfaceset.csv'
FIELD_NAMES = ['path', 'xmin', 'ymin', 'xmax', 'ymax', 'id', 'mask']


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

    def save(self, shapes, targetFile=TARGET_FILE, path_dictionary=None, ):
        lines = []
        path = self.localImgPath
        i, start, end = 0, -1, -1
        if not os.path.exists(targetFile):
            return
        with open(targetFile, mode='r', encoding='utf-8') as r:
            reader = csv.DictReader(r, self.fieldnames)
            for row in reader:
                if os.path.normpath(row['path']) != path:
                    lines.append(row)
                else:
                    if start == -1:
                        start = i
                        end = i
                    else:
                        end += 1
                i += 1
        for shape in shapes:
            p = []
            p.append(round(shape.points[0].x()))
            p.append(round(shape.points[0].y()))
            p.append(round(shape.points[2].x()))
            p.append(round(shape.points[2].y()))

            lines.append(
                {'path': path, 'xmin': p[0], 'ymin': p[1], 'xmax': p[2], 'ymax': p[3],
                 'id': shape.userid, 'mask': shape.mask})

        os.remove(targetFile)
        with open(targetFile, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, self.fieldnames)
            writer.writerows(lines)
        return start, end


class OneFileReader:
    def __init__(self, ):
        self.shapes = None

        self.fieldnames = FIELD_NAMES

    def loadShapes(self, imagePath, targetFile=TARGET_FILE, prepath=None):
        prepath = os.path.normpath(prepath)
        shapes = []
        if not os.path.exists(targetFile):
            return
        if len(prepath)> 2 :
            if prepath in imagePath:
                imagePath = imagePath.split(prepath)[-1][1:]

        with open(targetFile, mode='r', encoding='utf-8') as r:
            reader = csv.DictReader(r, self.fieldnames)
            i = 0
            for row in reader:
                if row['path'] == imagePath:
                    shapes.append(
                        [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), int(row['id']),
                         int(row['mask'])])
        return shapes
