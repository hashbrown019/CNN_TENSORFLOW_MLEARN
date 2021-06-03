import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import json
import pathlib

batch_size = 100
img_height = 180
img_width = 180
SRC_MODEL = 'models/model.h5'
SRC_DATASETS = "datasets/"
SRC_CLASSES = "models/classes"
epochs=5
gpu_id = 0

def init_gpu():
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = "PCI\\VEN_10DE&DEV_1D11&SUBSYS_14151025&REV_A1"

	config = tf.compat.v1.ConfigProto()
	sess = tf.compat.v1.Session(config=config)

	with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
		print(tf.DeviceSpec(device_type="GPU", device_index=gpu_id))

def training():
	data_dir = pathlib.Path(SRC_DATASETS) 
	image_count = len(list(data_dir.glob('*/*.jpg')))

	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		data_dir,
		validation_split=0.2,
		subset="training",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size)

	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
		data_dir,
		validation_split=0.2,
		subset="validation",
		seed=123,
		image_size=(img_height, img_width),
		batch_size=batch_size)

	class_names = train_ds.class_names
	AUTOTUNE = tf.data.AUTOTUNE

	train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

	normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

	normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
	image_batch, labels_batch = next(iter(normalized_ds))
	first_image = image_batch[0]

	num_classes = len(class_names)+1

	model = Sequential([
		layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
		layers.Conv2D(16, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Conv2D(32, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Conv2D(64, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Flatten(),
		layers.Dense(128, activation='relu'),
		layers.Dense(num_classes)
	])
	
	model.compile(optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy'])
	model.summary()
	history = model.fit(train_ds,validation_data=val_ds,epochs=epochs)
	print("-------------------------")
	print(history)
	model.save(SRC_MODEL)
	open(SRC_CLASSES,'w').write(json.dumps(class_names))

# =============================================================

def compares():
	model = keras.models.load_model(SRC_MODEL)
	classes = json.loads(open(SRC_CLASSES,'r').read())
	img = keras.preprocessing.image.load_img( "test/1.jpg", target_size=(img_height, img_width))
	img_array = keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch

	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])

	print("----------------------------------------------------------------")
	print(
	    "This image most likely belongs to |[{}]| with a |[{:.2f}]| percent confidence."
	    .format(classes[np.argmax(score)], 100 * np.max(score))
	)

# training()
compares()
