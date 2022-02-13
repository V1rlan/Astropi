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
import csv
from datetime import datetime


# Given the interpreter and labels it returns a dictionary with the results
def dict_of_scores(interpreter, labels):

        results = dict()
        values = classify.get_scores(interpreter)
        for ii in range(classify.num_classes(interpreter)):
                results[labels[ii]] = values[ii]
        return results

""" 
 Adds a line to the cvs file with the relative image, the longitude, latitude 
 and time of capture with the interpeter results
"""
def csv_addline(count, results):
        location = ISS.coordinates()
        list_writer = csv.writer(list_file, delimiter=",")
        list_writer.writerow(["Coast_{}".format(count), location.longitude, location.latitude, datetime.now(), results['coast']])

try:
    # Time start of the script
    start_time = datetime.now()

    # csv file creation and setup
    list_file = open("csv_file.csv", "w")
    list_writer = csv.writer(list_file, delimiter=",")
    list_writer.writerow(['Coast number','Longitude','Latitude','Time','Coast score'])
except:
    exit()

# Relative paths' setup
script_dir = Path(__file__).parent.resolve()
model_dir = script_dir/'models'
model_file = model_dir/'astropi-coast-vs-notcoast.tflite' # Path of the interpreter
label_file = model_dir/'labels.txt'                                # Label's path
image_file = script_dir/'temp'/'temp.jpg'                        # Path and name of the image to classify
final_path = script_dir/'coast'                                   # Folder of a coast image

# TPU's setup
interpreter = make_interpreter(f"{model_file}") # model_file is not a string, f"{model_file}" casts it
interpreter.allocate_tensors()
labels = read_label_file(label_file)

# Camera's setup
camera = PiCamera()
camera.resolution = (2592, 1944)
camera.framerate = 50

# Setup of last variables
count = 0                 # Counter coasts
now_time = datetime.now() # Current time


# main while
while (now_time < start_time + timedelta(minutes=1)):
        try:
                # Capture the image
                camera.capture(f"{image_file}")

                # Gets the resolution at which the interpreter works and sets the image to such resolution
                size = common.input_size(interpreter)
                image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

                # Image classification
                common.set_input(interpreter, image)
                interpreter.invoke()

                # Output of the interpreter
                results = dict_of_scores(interpreter, labels)

                # This if decides if the image is a coast or not
                if (results['coast'] > -0.2):

                        csv_addline(count, results)

                        # Image_file path is changed to a specific folder (final_path)
                        image_file.replace(script_dir / final_path /'Coast{}.jpg'.format(count))

                        # count increased by one for a coast image has been classified
                        count += 1
                else:
                        # The image is not a coast so it is eliminated
                        image_file.unlink(missing_ok=True)

                # Wait 5 seconds
                sleep(5)

                # now_time variable is refreshed with the current time
                now_time = datetime.now()
        except Exception as e:
                # This handle all exception and save them on the csv file
                now_time = datetime.now()
                list_writer.writerow([f'{e.__class__.__name__}: {e}',now_time])
                sleep(5)

# The csv file is closed
list_file.close()
