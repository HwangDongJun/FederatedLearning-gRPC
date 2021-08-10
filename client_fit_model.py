from __future__ import absolute_import, division, print_function, unicode_literals

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class learning_fit(object):
	def __init__(self):
		self.temp = ""

	def image_generator(self):
		image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
																	rotation_range=15,
																	horizontal_flip=True,
																	brightness_range=[0.7, 1.0])
		image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
		return image_gen_train, image_gen_val

	def gen_train_val_data(self, bs=16, ims=(224, 224), cn=np.array(['0', '1', '2', '3', '4'])):
		gen_train, gen_val = self.image_generator()

		train_data_dir = os.path.abspath('/home/dnlab/federated_grpc/data_balance/train/')
		train_data_gen = gen_train.flow_from_directory(directory=str(train_data_dir),
														batch_size=bs,
														color_map='rgb',
														shuffle=True,
														target_size=ims,
														classes=list(cn))
		return train_data_gen

	def manage_train(self, cr, params): # cr:current_round
		get_weights = list()
		if cr != 1: # 첫 학습
			
		else:
			
