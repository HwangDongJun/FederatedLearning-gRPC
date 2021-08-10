import pickle
import numpy as np
import PIL.Image as Image
from PIL import ImageFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

class evaluate_LocalModel(object):
	def __init__(self, batch_size, image_size, class_names):
		self.batch_size = batch_size
		self.image_shape = (image_size, image_size)
		self.class_names = class_names

	def image_generator(self):
		return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

	def gen_test_val_data(self):
		gen_val = self.image_generator()
		val_data_dir = os.path.abspath('/home/dnlab/federated_grpc/data_balance/test/')
		val_data_gen = gen_val.flow_from_directory(directory=str(val_data_dir),
				        batch_size=self.batch_size,
				        color_mode='rgb',
				        shuffle=False,
				        target_size=self.image_shape,
				        classes=list(self.class_names))
		return val_data_gen

	def buildGlobalModel(self, channels, lr):
		model = tf.keras.Sequential([
				layers.InputLayer(input_shape=self.image_shape+(channels,)),
				layers.Conv2D(20, (5, 5), strides=(1, 1), activation='relu'),
				layers.MaxPooling2D((2, 2)),
				layers.Conv2D(50, (5, 5), strides=(1, 1), activation='relu'),
				layers.MaxPooling2D((2, 2)),
				layers.Conv2D(30, (5, 5), strides=(1, 1), activation='relu'),
				layers.Conv2D(20, (5, 5), strides=(1, 1), activation='relu'),
				layers.Dropout(0.5),
				layers.Flatten(),
				layers.Dense(10, activation='relu'),
				layers.Dense(5, activation='softmax')
		])

		model.compile(
				optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
				loss='categorical_crossentropy',
				metrics=['accuracy'])

		return model

	#def saved_model(self, localmodel):
		#localmodel.save("./saved_model/")

	def get_weights(self, localmodel):
		return pickle.dumps(localmodel.get_weights())

	def train_model_tosave(self, localmodel):
		print("### Start model test ###")
		gen_val_data = self.gen_test_val_data()
		result = localmodel.evaluate(gen_val_data)
		return dict(zip(localmodel.metrics_names, result))
