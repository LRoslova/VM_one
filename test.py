
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# CIFAR-10 classes
class_names = {
    0: "alcohol",
    1: "drugs",
    2: "ordinary",
    3: "porn",
    4: "weapon"
}

img_height = 180
img_width = 180

model = keras.models.load_model('mymodel-v2.h5')

test_url = "https://wl-adme.cf.tsp.li/resize/728x/JPG/34f/691/c443c555b9ae8b9b4aff7b1013.JPG"
test_path = tf.keras.utils.get_file('test_eat', origin=test_url)
img = tf.keras.utils.load_img(
    test_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

