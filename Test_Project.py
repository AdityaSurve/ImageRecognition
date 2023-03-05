from keras.models import model_from_json
from pathlib import Path
import numpy as np
import tensorflow
class_labels=[
    "Planes",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Boat",
    "Truck"
]
file=Path("model_structure.json")
model_structure=file.read_text()
model=model_from_json(model_structure)
model.load_weights("model_weights.h5")
img=tensorflow.keras.utils.load_img("temp.png",target_size=(32,32))
image_to_test=tensorflow.keras.utils.img_to_array(img)
list_of_images=np.expand_dims(image_to_test,axis=0)
results=model.predict(list_of_images)
single_result=results[0]
most_likely_class_index=int(np.argmax(single_result))
class_likelihood=single_result[most_likely_class_index]
class_label=class_labels[most_likely_class_index]

print("Image = {}-Likelihood: {:2f}".format(class_label, class_likelihood))