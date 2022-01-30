from pathlib import Path
from PIL import Image

from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

from picamera import PiCamera
from datetime import datetime, timedelta
from time import sleep

from orbit import ISS
import os
import csv
from datetime import datetime


# Dato l'interpreter e i labels restituisce il dizionario con i risultati
def dict_of_scores(interpreter, labels):
        risultati = dict()
        valori = classify.get_scores(interpreter)
        for ii in range(classify.num_classes(interpreter)):
                risultati[labels[ii]] = valori[ii]
        return risultati

def path_changer_coast(source_path, final_path):
    # replace(path sorgente, path finale), spostando rinonima
    os.replace(source_path, "{}/Costa{}.jpg".format(final_path,cont))

def path_changer_not_coast(source_path):
    # Rimuovo il file
    os.remove(source_path)

# Aggiunge riga al file csv
def csv_addline(n_foto, risultati):
        location = ISS.coordinates()
        list_writer = csv.writer(list_file, delimiter=",")
        list_writer.writerow(["Costa_{}".format(cont), location.longitude,location.latitude, datetime.now(), risultati['costa']])

#Tempo di inizio script
start_time = datetime.now()

#Setup dei path relativi
script_dir = Path(__file__).parent.resolve()
model_file = script_dir/'models/astropi-costa-vs-nocosta.tflite' #nome del modello allenato
data_dir = script_dir/'data' 
label_file = data_dir/'labels.txt' # path dei label
image_file = script_dir/'tests'/'temp.jpg' # nome della immagine da classificare 

# Setup della TPU
interpreter = make_interpreter(f"{model_file}") #model file non e una stringa, f"{model_file}" lo rende una stringa
interpreter.allocate_tensors()
labels = read_label_file(label_file)

#Setup delal camera
camera = PiCamera()
camera.resolution = (2592, 1944)
camera.framerate = 50

# Setup
cont = 0        # Contatore foto costa
list_file = open("csv_file.csv", "w")
list_writer = csv.writer(list_file, delimiter=",")
list_writer.writerow(['Num Costa','Longitudine','Latitudine','orario','score costa'])
now_time = datetime.now()

while (now_time < start_time + timedelta(minutes=2)):
        camera.capture(f"{image_file}")

        # Add the following lines to find out the dimensions used for the model you have retrained, and then set the size of your test image to the same dimensions, using PIL.
        size = common.input_size(interpreter)
        image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

        # Classificazione dell-immagine
        common.set_input(interpreter, image)
        interpreter.invoke()
        classes = classify.get_classes(interpreter)

        # Output
        risultati = dict_of_scores(interpreter, labels)

        if (risultati['costa'] > 0.2):
                csv_addline(cont, risultati)

                path_changer_coast(f"{image_file}", f"{data_dir}")
                cont += 1
        else:
                os.remove(f"{image_file}")

        sleep(5)        
        now_time = datetime.now()

list_file.close()
