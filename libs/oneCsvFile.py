import os
import codecs
from libs.constants import DEFAULT_ENCODING
import csv
from libs.utils import convertPointsToXY

TARGET_FILE = 'localfaceset.csv'
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

    def save(self, shapes, targetFile=TARGET_FILE, path_dictionary=None):
        lines = []
        path = None
        for shape in shapes:
            path = path_dictionary[shape.id]
            break
        if not os.path.exists(targetFile):
            return
        with open(targetFile, mode='r', encoding='utf-8') as r:
            reader = csv.DictReader(r, self.fieldnames)
            for row in reader:
                if os.path.normpath(row['path']) != path:
                    lines.append(row)
                else:
                    for shape in shapes:
                        p = []
                        p.append(round(shape.points[0].x()))
                        p.append(round(shape.points[0].y()))
                        p.append(round(shape.points[2].x()))
                        p.append(round(shape.points[2].y()))

                        lines.append(
                            {'path': path, 'xmin': p[0], 'ymin': p[1], 'xmax': p[2], 'ymax': p[3],
                             'id': shape.userid, })
        os.remove(targetFile)
        with open(targetFile, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, self.fieldnames)
            writer.writerows(lines)



class OneFileReader:
    def __init__(self, ):
        self.shapes = None

        self.fieldnames = FIELD_NAMES

    def loadShapes(self, imagePath, targetFile=TARGET_FILE,  ):
        shapes = []
        if not os.path.exists(targetFile):
            return
        if 'labelImg-master' in imagePath:
            imagePath = os.path.normpath(imagePath.split('labelImg-master')[-1][1:])

        with open(targetFile, mode='r', encoding='utf-8') as r:
            reader = csv.DictReader(r, self.fieldnames)
            i = 0
            for row in reader:
                if row['path'] == imagePath:

                    shapes.append([int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['id']])
        return shapes
