from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import pickle
import random
import numpy as np
import collections
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
		self.data_size = None;
		self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

		self.random_train_images = np.array([])
		self.random_train_labels = np.array([])
		self.select_test_images = np.array([])
		self.select_test_labels = np.array([])
		self.random_choice_list = list()

		self.train_data = None
		self.train_label = None
		self.test_data = None
		self.test_label = None

	def generate_datasets(self):		
		(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()
		
		#hete_count_choice = random.choice([0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5])
		#sample_hete_list = random.sample([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], hete_count_choice)
		'''
		sample_hete_list = random.sample([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 4)
		if self.random_choice_list == list():
			#self.random_choice_list = list(set([random.randint(10, 59999) for r in range(random.choice(list(set([random.randint(10, 49999) for r in range(30)]))))]))
			self.random_choice_list = list(set([random.randint(12000, 17999) for r in range(10000)]))

		with open('./save_random_data/random_choice13.pickle', 'wb') as fw:
			pickle.dump(self.random_choice_list, fw)
		with open('./save_random_data/hete_list13.pickle', 'wb') as fw:
			pickle.dump(sample_hete_list, fw)
		'''
		with open('./save_random_data/random_choice13.pickle', 'rb') as fr:
			self.random_choice_list = pickle.load(fr)
		with open('./save_random_data/hete_list13.pickle', 'rb') as fr:
			sample_hete_list = pickle.load(fr)


		for i, rcl in enumerate(self.random_choice_list):
			if int(train_labels[rcl]) not in sample_hete_list:
				if len(self.random_train_images) == 0:
					self.random_train_images = np.expand_dims(train_images[rcl], axis=0)
					self.random_train_labels = train_labels[rcl]
				else:
					self.random_train_images = np.append(self.random_train_images, np.expand_dims(train_images[rcl], axis=0), axis=0)
					self.random_train_labels = np.append(self.random_train_labels, train_labels[rcl])
	
		self.data_size = dict(collections.Counter(self.random_train_labels))
		
		# test images
		for i in range(10000):
			#if int(test_labels[i]) not in sample_hete_list:
			if len(self.select_test_images) == 0:
				self.select_test_images = np.expand_dims(test_images[i], axis=0)
				self.select_test_labels = test_labels[i]
			else:
				self.select_test_images = np.append(self.select_test_images, np.expand_dims(test_images[i], axis=0), axis=0)
				self.select_test_labels = np.append(self.select_test_labels, test_labels[i])

		#train_images = train_images.reshape((train_len, 28, 28, 1))
		train_images = self.random_train_images.reshape((len(self.random_train_labels), 28, 28, 1))
		#test_images = test_images.reshape((10000, 28, 28, 1))
		test_images = self.select_test_images.reshape((len(self.select_test_labels), 28, 28, 1))
	
		train_images, test_images = train_images / 255.0, test_images / 255.0

		#return train_images, train_labels, test_images, test_labels
		return train_images, self.random_train_labels, test_images, self.select_test_labels
			

	def generate_model(self):
		model = models.Sequential()
		model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
		model.add(layers.MaxPooling2D((2, 2)))
		model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2, 2)))
		model.add(layers.Flatten())
		model.add(layers.Dense(64, activation='relu'))
		model.add(layers.Dense(10, activation='softmax'))

		return model

	def build_model(self):
		new_model = self.generate_model()
		new_model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
		return new_model
	
	def train_model_tosave(self, params, rounds, cn):
		earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
		logdir = f"send_logs/logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}-{cn}-{rounds}"
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

		if params == None: # 데이터수를 라운드지날때마다 늘리지 않기 위함
			self.train_data, self.train_label, self.test_data, self.test_label = self.generate_datasets()
		local_model.fit(self.train_data, self.train_label, epochs=self.epochs, callbacks=[earlystopping_callback, tensorboard_callback])
		acc_loss = local_model.evaluate(self.test_data, self.test_label, verbose=2)

		return local_model, acc_loss[1], acc_loss[0]

	def manage_train(self, params=None, cr=None, cn=None): # cr:current_round
		STIME = time.time()
		print(f"### Model Training - Round: {cr} ###")
		if self.params == list():
			return []

		if params != None:
			params = pickle.loads(params)
		lmodel, acc, loss = self.train_model_tosave(params, cr, cn)
		params = lmodel.get_weights()
		print("### Save model weight to ./saved_weight/ ###")
		with open('./saved_weight/weights13.pickle', 'wb') as fw:
			pickle.dump(params, fw)

		ETIME = time.time() - STIME
		return acc, loss, ETIME, self.data_size, self.class_names
