import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from keras import layers
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers import MaxPooling2D, GlobalMaxPooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow_addons as tfa


BATCH_SIZE = 8
FREEZE_LAYERS = 2

train_dir = './training_dataset'
train_ds = tf.keras.utils.image_dataset_from_directory(train_dir, image_size = (224, 224), batch_size = BATCH_SIZE)
val_dir = './validation_dataset'
val_ds = tf.keras.utils.image_dataset_from_directory(val_dir, image_size = (224, 224), batch_size = BATCH_SIZE)

net = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = net.output
x = Flatten()(x)
x = Dropout(0.5)(x)
output_layer = Dense(1, activation='sigmoid')(x)
model = Model(inputs=net.input, outputs=output_layer)

# for layer in model.layers[:FREEZE_LAYERS]:
#     layer.trainable = False
# for layer in model.layers[FREEZE_LAYERS:]:
#     layer.trainable = True

loss_function = tf.keras.losses.BinaryCrossentropy()
# loss_function = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.4, gamma=1.0)
optimizer = tf.keras.optimizers.Adam(learning_rate = 8e-5)
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

model.summary()
model.fit(train_ds, validation_data = val_ds, epochs = 10)

model.save('binary.model')