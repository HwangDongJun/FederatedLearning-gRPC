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

os.environ["CUDA_VISIBLE_DEVICES"] = ""

class learning_fit(object):
	def __init__(self, mt, ec, bs, pr, cr):
		'''
		gpus = tf.config.experimental.list_physical_devices('GPU')
		if gpus:
			try:
				# Currently, memory growth needs to be the same across GPUs
				#for gpu in gpus:
				#	tf.config.experimental.set_memory_growth(gpu, True)
				tf.config.experimental.set_virtual_device_configuration(
					gpus[0],
					[tf.config.experimental.VirtualDeviceConfiguration(memroy_limit=639)])
				logical_gpus = tf.config.experimental.list_logical_devices('GPU')
				print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
			except RuntimeError as e:
				# Memory growth must be set before GPUs have been initialized
				print(e)
		'''
		
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
		
		#sample_hete_list = random.sample([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 6)
		#sample_hete_list = random.sample([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], random.choice(list(range(2, 10))))
		sample_hete_list = [1, 3, 4, 6, 7, 8]
		if self.random_choice_list == list():
			#self.random_choice_list = list(set([random.randint(10, 59999) for r in range(random.choice(list(set([random.randint(10, 39999) for r in range(30)]))))]))
			#self.random_choice_list = list(set([random.randint(48000, 53999) for r in range(10000)]))
			self.random_choice_list = list(range(48000, 53999))
		'''	
		with open('./save_random_data/random_choice9_newloss.pickle', 'wb') as fw:
			pickle.dump(self.random_choice_list, fw)
		with open('./save_random_data/hete_list9_newloss.pickle', 'wb') as fw:
			pickle.dump(sample_hete_list, fw)
		
		with open('./save_random_data/random_choice9.pickle', 'rb') as fr:
			self.random_choice_list = pickle.load(fr)
		with open('./save_random_data/hete_list9.pickle', 'rb') as fr:
			sample_hete_list = pickle.load(fr)
		'''	
		for i, rcl in enumerate(self.random_choice_list):
			if int(train_labels[rcl]) not in sample_hete_list:
				if len(self.random_train_images) == 0:
					self.random_train_images = np.expand_dims(train_images[rcl], axis=0)
					self.random_train_labels = train_labels[rcl]
				else:
					self.random_train_images = np.append(self.random_train_images, np.expand_dims(train_images[rcl], axis=0), axis=0)
					self.random_train_labels = np.append(self.random_train_labels, train_labels[rcl])
	
		# new data --> class
		self.random_choice_list = list()
		sample_hete_list = [0, 1, 2, 3, 5, 6, 7, 8]
		if self.random_choice_list == list():
			self.random_choice_list = list(range(24300, 32500))
			
		for i, rcl in enumerate(self.random_choice_list):
			if int(train_labels[rcl]) not in sample_hete_list:
				self.random_train_images = np.append(self.random_train_images, np.expand_dims(train_images[rcl], axis=0), axis=0)
				self.random_train_labels = np.append(self.random_train_labels, train_labels[rcl])
		# new data --> class (small)
		self.random_choice_list = list()
		sample_hete_list = [0, 1, 2, 3, 4, 5, 7, 8, 9]
		if self.random_choice_list == list():
			self.random_choice_list = list(range(36000, 40999))
			
		for i, rcl in enumerate(self.random_choice_list):
			if int(train_labels[rcl]) not in sample_hete_list:
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
		model.add(layers.Conv2D(32, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2, 2)))
		model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		model.add(layers.Conv2D(64, (3, 3), activation='relu'))
		model.add(layers.MaxPooling2D((2, 2)))
		model.add(layers.Flatten())
		model.add(layers.Dense(256, activation='relu'))
		model.add(layers.Dense(10, activation='softmax'))

		return model

	def build_model(self):
		new_model = self.generate_model()
		new_model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])
		return new_model
	
	def train_model_tosave(self, params, rounds, cn, breakin=False):
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

		if params == None or breakin: # 데이터수를 라운드지날때마다 늘리지 않기 위함
			self.train_data, self.train_label, self.test_data, self.test_label = self.generate_datasets()
			
		acc_loss = local_model.evaluate(self.test_data, self.test_label, verbose=2)
		print(f"### {acc_loss} ###")
		
		hist = local_model.fit(self.train_data, 
					self.train_label, 
					epochs=self.epochs, 
					validation_data=(self.test_data, self.test_label),
					callbacks=[earlystopping_callback, tensorboard_callback])
		#acc_loss = local_model.evaluate(self.test_data, self.test_label, verbose=2)

		# test sample count
		predictions = local_model.predict(self.test_data)
		probability_wrong_count = dict()
		for i, pr in enumerate(predictions):
			if self.test_label[i] != np.argmax(pr):
				if str(self.test_label[i]) not in probability_wrong_count:
					probability_wrong_count[str(self.test_label[i])] = 1
				else:
					probability_wrong_count[str(self.test_label[i])] += 1
					
		print("###################################")
		print(f"### {probability_wrong_count} ###")
		print("###################################")
		#mean_class_probability = dict()
		#for pc in probability_wrong_count.keys():
		#	mean_class_probability[pc] = sum(probability_count[pc])/len(probability_count[pc])
		pred_wrong = sum(probability_wrong_count.values()) / len(self.test_data)

		#return local_model, acc_loss[1], acc_loss[0], hist.history['loss'][-1], mean_class_probability
		#return local_model, hist.history['val_accuracy'][-1], hist.history['val_loss'][-1], hist.history['loss'][-1], pred_wrong
		return local_model, acc_loss[1], acc_loss[0], hist.history['loss'][-1], pred_wrong

	def manage_train(self, params=None, cr=None, cn=None, state=None): # cr:current_round
		STIME = time.time()
		print(f"### Model Training - Round: {cr} ###")
		if self.params == list():
			return []

		if params != None:
			params = pickle.loads(params)
		if state != None:
			lmodel, acc, loss, tloss, mcp = self.train_model_tosave(params, cr, cn, True)	
		else:
			lmodel, acc, loss, tloss, mcp = self.train_model_tosave(params, cr, cn)
		params = lmodel.get_weights()
		print("### Save model weight to ./saved_weight/ ###")
		with open('./saved_weight/weights9.pickle', 'wb') as fw:
			pickle.dump(params, fw)

		ETIME = time.time() - STIME
		return acc, loss, tloss, ETIME, self.data_size, self.class_names, mcp
