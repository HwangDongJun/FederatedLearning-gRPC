from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import tensorflow_hub as hub

os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

	def test_image_generator(self):
		test_image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
		return test_image_gen_val

	def gen_train_val_data(self, ims=(96, 96), cn=np.array(['0', '1', '2', '3', '4']), crr=None):
		gen_train, gen_val = self.image_generator()

		train_data_dir = None
		if crr >= 160 and crr < 180: # 10
			train_data_dir = os.path.abspath('/home/dnlab/FederatedLearning-gRPC/client/save_data/client10_211110/')
		elif crr >= 180: # 11
			train_data_dir = os.path.abspath('/home/dnlab/FederatedLearning-gRPC/client/save_data/client10_211111/')
		train_data_gen = gen_train.flow_from_directory(directory=str(train_data_dir),
														batch_size=self.batch_size,
														color_mode='rgb',
														shuffle=True,
														target_size=ims,
														classes=list(cn))
		return train_data_gen

	def gen_test_val_data(self, ims=(96, 96), cn=np.array(['0', '1', '2', '3', '4'])):
		gen_val = self.test_image_generator()
		val_data_dir = os.path.abspath('/home/dnlab/FederatedLearning-gRPC/client/save_data/test/')
		val_data_gen = gen_val.flow_from_directory(directory=str(val_data_dir),
								batch_size=self.batch_size,
								color_mode='rgb',
								shuffle=False,
								target_size=ims,
								classes=list(cn))
		return val_data_gen

	def change_model_layers(self, model):
		'''
		base_input = model.layers[0].input
		base_output = model.layers[-2].output
		
		# 지금은 아니지만 서버와 클라이언트가 학습 모델에 대한 정보를 알 수 있는 방법도 코드 구현해야함
		final_output = layers.Dense(32)(base_output)
		final_output = layers.Activation('relu')(final_output)
		#final_output = layers.Dense(64)(final_output)
		#final_output = layers.Activation('relu')(final_output)
		final_output = layers.Dense(5, activation='softmax')(final_output)

		return keras.Model(inputs=base_input, outputs=final_output)
		'''
		base_model = tf.keras.Sequential([
				model,
				layers.Dense(5, activation='softmax')
		])
		return base_model

	def build_model(self, weight=None):
		print(self.model_type)
		if self.model_type == "mobilenet_v2":
			#model = tf.keras.applications.MobileNetV2()
			model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/5"
			model = hub.KerasLayer(model_url, input_shape=(96, 96, 3))

			if weight == None:
				model.trainable = False
			else:
				model.trainable = True

			new_model = self.change_model_layers(model)
			if weight == None:
				new_model.compile(
						optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
						loss="categorical_crossentropy", 
						metrics=["accuracy"])
			else:
				new_model.compile(
						optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
						loss="categorical_crossentropy",
						metrics=["accuracy"])
			return new_model
		else:
			return None # another model?

	def train_model_tosave(self, params, cr, breakin=False):
		earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
		logdir = f"send_logs/logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self.round}"
		tensorboard_callback = tf.keras.callbacks.TensorBoard(
														log_dir=logdir,
														histogram_freq=1,
														write_graph=True,
														update_freq='epoch',
														profile_batch=2,
														embeddings_freq=1)

		local_model = self.build_model(params)
		if params != None:
			local_model.set_weights(params)
		#else:
		#	local_model.set_weights(self.params)
		
		gen_val_data = self.gen_test_val_data()
		acc_loss = local_model.evaluate(gen_val_data)

		gen_train_data = self.gen_train_val_data(crr=cr)
		
		# label count
		train_class_data_count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
		for gtd in list(gen_train_data.labels):
			train_class_data_count[int(gtd)] += 1
		
		hist = None
		if params == None:
			#local_model.fit_generator(gen_train_data, epochs=self.epochs, callbacks=[earlystopping_callback, tensorboard_callback])
			hist = local_model.fit_generator(gen_train_data, epochs=self.epochs, callbacks=[earlystopping_callback])
		else:
			#local_model.fit_generator(gen_train_data, epochs=self.epochs*2, callbacks=[earlystopping_callback, tensorboard_callback])
			hist = local_model.fit_generator(gen_train_data, epochs=self.epochs*2, callbacks=[earlystopping_callback])

		# test sample count
		predictions = local_model.predict(gen_val_data)
		predicted_id = np.argmax(predictions, axis=-1)
		label_id = list(gen_val_data.labels)
		probability_wrong_count = dict()
		for i, pr in enumerate(predicted_id):
			if label_id[i] != predicted_id[i]:
				if str(label_id[i]) not in probability_wrong_count:
					probability_wrong_count[str(label_id[i])] = 1
				else:
					probability_wrong_count[str(label_id[i]	)] += 1
		#mean_class_probability = dict()
		#for pc in probability_wrong_count.keys():
		#	mean_class_probability[pc] = sum(probability_count[pc])/len(probability_count[pc])
		pred_wrong = sum(probability_wrong_count.values()) / len(label_id)


		# Export model
		#export_path = "?"
		#local_model.save(export_path, save_format="tf")
			
		return local_model, acc_loss, hist.history['loss'][-1], pred_wrong, train_class_data_count

	def manage_train(self, params=None, cr=None, cn=None, state=None): # cr:current_round
		STIME = time.time()
		print(f"### Model Training - Round: {cr} ###")
		if self.params == list():
			return []

		if params != None:
			params = pickle.loads(params)
		if state != None:
			lmodel, res_acc_loss, tloss, mcp, tcdc = self.train_model_tosave(params, cr, True)
		else:
			lmodel, res_acc_loss, tloss, mcp, tcdc = self.train_model_tosave(params, cr)
		params = lmodel.get_weights()
		print("### Save model weight to ./saved_weight/ ###")
		with open('./saved_weight/weights10.pickle', 'wb') as fw:
			pickle.dump(params, fw)
		ETIME = time.time() - STIME
		acc = dict(zip(lmodel.metrics_names, res_acc_loss))['accuracy']
		loss = dict(zip(lmodel.metrics_names, res_acc_loss))['loss']
		#data_size = {0: 365, 1: 0, 2: 54, 3: 130, 4: 194}
		#return acc, loss, ETIME, data_size, ['0', '1', '2', '3', '4']
		return acc, loss, tloss, ETIME, tcdc, ['0', '1', '2', '3', '4'], mcp
