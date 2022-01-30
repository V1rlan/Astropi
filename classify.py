from pathlib import Path
from PIL import Image

from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

from picamera import PiCamera
from datetime import datetime, timedelta
from time import sleep

# Dato l'interpreter e i labels restituisce il dizionario con i risultati
def dict_of_scores(interpreter, labels):
        risultati = dict()
        valori = classify.get_scores(interpreter)
        for ii in range(classify.num_classes(interpreter)):
                risultati[labels[ii]] = valori[ii]
        return risultati

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
        print('foto nuova')
        print(risultati)
        sleep(5)        
        now_time = datetime.now()
