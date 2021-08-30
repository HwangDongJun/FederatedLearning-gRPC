from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from datetime import datetime


class learning_fit(object):
	def __init__(self, mt, ec, bs, pr, cr):
		self.model_type = mt
		self.epochs = ec
		self.batch_size = bs
		self.params = pickle.loads(pr)
		self.round = cr

		self.mnist = keras.datasets.mnist
		self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

	def generate_datasets(self):
		(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

		train_images = train_images.reshape((60000, 28, 28, 1))
		test_images = test_images.reshape((10000, 28, 28, 1))

		train_images, test_images = train_images / 255.0, test_images / 255.0

		return train_images, train_labels, test_images, test_labels

	def generate_model(self):
		model = models.Sequential()
		model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
		model.add(layers.MaxPooling2D((2, 2)))
		model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2, 2)))
		model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		model.add(layers.Flatten())
		model.add(layers.Dense(64, activation='relu'))
		model.add(layers.Dense(10, activation='softmax'))

		return model

	def build_model(self):
		new_model = self.generate_model()
		new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
		return new_model
	
	def train_model_tosave(self, params):
		earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
		logdir = f"send_logs/logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.round}"
		tensorboard_callback = tf.keras.callbacks.TensorBoard(
					log_dir=logdir,
					histogram_freq=1,
					write_graph=True,
					update_freq='epoch',
					profile_batch=2,
					embeddings_freq=1)

		local_model = self.build_model()
		if params != None:
			local_model.set_weights(params)
		else:
			local_model.set_weights(self.params)

		train_data, train_label, test_data, test_label = self.generate_datasets()
		local_model.fit(train_data, train_label, epochs=self.epochs, callbacks=[earlystopping_callback, tensorboard_callback])

		return local_model

	def manage_train(self, params=None, cr=None): # cr:current_round
		print(f"### Model Training - Round: {cr} ###")
		if self.params == list():
			return []

		if params != None:
			params = pickle.loads(params)
		lmodel = self.train_model_tosave(params)
		params = lmodel.get_weights()
		print("### Save model weight to ./saved_weight/ ###")
		with open('./saved_weight/weights.pickle', 'wb') as fw:
			pickle.dump(params, fw)
