from os import listdir
from os.path import isfile, join
import os
import csv

def concatcsv(path=None):
    if os.path.exists(join(path, 'all.csv')):
        os.remove(join(path, 'all.csv'))
    files = [f for f in listdir(path) if isfile(join(path, f))]
    header = True
    for i in range(len(files)):
        if files[i].split(".")[-1] == 'csv':
            if files[i].split(".")[0] != 'all':
                with open(join(path, files[i]), mode='r')as r:
                    readerField = ['Id', 'xmin', 'ymin', 'xmax', 'ymax']
                    reader = csv.DictReader(r, readerField)

                    with open(join(path, 'all.csv'), mode='a', newline='') as f:
                        writerField = ['path', 'Id', 'xmin', 'ymin', 'xmax', 'ymax']
                        writer = csv.DictWriter(f, writerField)
                        if header :
                            writer.writeheader()
                            header = False
                        for raw in reader:
                            writer.writerow({'path': files[i][0:-4], 'Id': raw['Id'], 'xmin': raw['xmin'],
                                             'ymin': raw['ymin'], 'xmax': raw['xmax'], 'ymax': raw['ymax']})


concatcsv('F:/picture/5')