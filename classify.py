from pathlib import Path
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

#salva il path dello script
script_dir = Path(__file__).parent.resolve()

#salva i path delle informazioni
model_file = script_dir/'models/astropi-costa-vs-nocosta.tflite' #nome del modello allenato
data_dir = script_dir/'data' 
label_file = data_dir/'labels.txt' # path dei label
image_file = script_dir/'tests'/'sicosta.jpg' # nome della immagine da classificare 

# Setup della TPU
interpreter = make_interpreter(f"{model_file}")
interpreter.allocate_tensors()

# Add the following lines to find out the dimensions used for the model you have retrained, and then set the size of your test image to the same dimensions, using PIL.
size = common.input_size(interpreter)
image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

# Classificazione dell-immagine
common.set_input(interpreter, image)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

# Output
labels = read_label_file(label_file)
for c in classes:
        print(f'{labels.get(c.id, c.id)} {c.score:.5f}')
