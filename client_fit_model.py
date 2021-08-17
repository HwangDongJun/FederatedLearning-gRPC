from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime


class learning_fit(object):
	def __init__(self, mt, ec, bs, pr, cr):
		self.model_type = mt
		self.epochs = ec
		self.batch_size = bs
		self.params = pickle.loads(pr)
		self.round = cr

	def image_generator(self):
		image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
																	rotation_range=15,
																	horizontal_flip=True,
																	brightness_range=[0.7, 1.0])
		image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
		return image_gen_train, image_gen_val

	def gen_train_val_data(self, ims=(224, 224), cn=np.array(['0', '1', '2', '3', '4'])):
		gen_train, gen_val = self.image_generator()

		train_data_dir = os.path.abspath('/home/dnlab/Downloads/data_balance/train/')
		train_data_gen = gen_train.flow_from_directory(directory=str(train_data_dir),
														batch_size=self.batch_size,
														color_mode='rgb',
														shuffle=True,
														target_size=ims,
														classes=list(cn))
		return train_data_gen

	def change_model_layers(self, model):
		base_input = model.layers[0].input
		base_output = model.layers[-2].output
		
		# 지금은 아니지만 서버와 클라이언트가 학습 모델에 대한 정보를 알 수 있는 방법도 코드 구현해야함
		final_output = layers.Dense(128)(base_output)
		final_output = layers.Activation('relu')(final_output)
		final_output = layers.Dense(64)(final_output)
		final_output = layers.Activation('relu')(final_output)
		final_output = layers.Dense(5, activation='softmax')(final_output)

		return keras.Model(inputs=base_input, outputs=final_output)

	def build_model(self):
		print(self.model_type)
		if self.model_type == "mobilenet_v2":
			model = tf.keras.applications.MobileNetV2()
			new_model = self.change_model_layers(model)
			new_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
			return new_model
		else:
			return None # another model?

	def train_model_tosave(self):
		earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
		logdir = f"send_logs/logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.round}"
		tensorboard_callback = tf.keras.callbacks.TensorBoard(
														log_dir=logdir,
														histogram_freq=1,
														write_graph=True,
														update_freq='epoch',
														profile_batch=2,
														embeddings_freq=1)

		local_model = self.build_model()
		local_model.set_weights(self.params)

		gen_train_data = self.gen_train_val_data()
		local_model.fit_generator(gen_train_data, epochs=self.epochs, callbacks=[earlystopping_callback, tensorboard_callback])

		# Export model
		#export_path = "?"
		#local_model.save(export_path, save_format="tf")
			
		return local_model

	def manage_train(self): # cr:current_round
		if self.params == list():
			return []
		lmodel = self.train_model_tosave()
		params = lmodel.get_weights()
		print("### Save model weight to ./saved_weight/ ###")
		with open('./saved_weight/weights.pickle', 'wb') as fw:
			pickle.dump(params, fw)
		return params
