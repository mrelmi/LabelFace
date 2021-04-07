How to use recommender mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~

First of all download model from  `here <https://www.dropbox.com/s/tj96fsm6t6rq8ye/model-r100-arcface-ms1m-refine-v2.zip?dl=0>` and put the model-r100-ii directory on './Boxes/'

install face-detection with "pip install face-detection"

your faceset csv should have this format ['id of path','xmin','ymin','xmax','ymax','id of face']

put the image paths csv and subjects csv in same folder with faceset.csv

consider the name of these 2 csv should be imagespath.csv and subjects.csv

imagespath.csv should have this format ['id of path' , 'path', 'get embeding from this path ?(flag) ']

and subjects.csv -> ['id of name ' , 'name']

after run the app :

you can click on "from csv" and choose your faceset csv (witch should have defined format)

a embs.npy will be made from that csv in long period that have the embedings dictionary of that csv . so be patient :)

please dont change imagespath and subjects manually carelessly.

load image , edit if it needs  and press save button (check under button , that should be on csv mode ,it changes with click )

you can set the Auto save mode in view dropdown and just click on next button ( or push 'D') but if you dont edit labels those shape will be saved  with unknown label and -1 for id

if you edit some image and save it again , it will be updated on faceset.csv

you can change constans on constants widget .

when founded 3 faces with more than second threshold it would be stop searching for similar faces .


