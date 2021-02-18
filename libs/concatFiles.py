from os import listdir
from os.path import isfile, join
import os
import csv


def concatcsv(path=None):
    if os.path.exists(join(path, 'all.csv')):
        os.remove(join(path, 'all.csv'))
    files = [f for f in listdir(path) if isfile(join(path, f))]
    indexes = []
    for i in range(len(files)):
        if files[i].split(".")[-1] == 'csv':
            if files[i].split(".")[0] != 'all':
                with open(join(path, files[i]), mode='r')as r:
                    readerField = ['Id', 'xcen', 'ycen', 'w', 'h']
                    reader = csv.DictReader(r, readerField)

                    with open(join(path, 'all.csv'), mode='a', newline='') as f:
                        writerField = ['path', 'Id', 'xcen', 'ycen', 'w', 'h']
                        writer = csv.DictWriter(f, writerField)

                        for raw in reader:
                            writer.writerow({'path': files[i][0:-4], 'Id': raw['Id'], 'xcen': raw['xcen'],
                                             'ycen': raw['ycen'], 'w': raw['w'], 'h': raw['w']})


