from __future__ import print_function

import os
import time
import random
import pickle

from threading import Thread
import psutil

from client_fit_model7 import learning_fit
#from client_fit_model_for_mnist7 import learning_fit

import grpc
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.transport_pb2 import Scalar, transportRequest, ReadyReq, UpdateReq, VersionReq, State
from common.transport_pb2_grpc import TransportServiceStub

class_for_learning = None

def request_parameter():
	update_request = [UpdateReq(type="P")]
	for ur in update_request:
		yield transportRequest(update_req=ur)

def request_ready():
	global client_name

	configuration = dict()
	configuration['current_round'] = Scalar(scint32=0)

	client_name = f'client{random.randint(1, 100000)}'
	ready_request = [ReadyReq(type='R', cname=client_name, state=State.ON, config=configuration)]
	for rr in ready_request:
		yield transportRequest(ready_req=rr)
	#return transportRequest(ready_req=ReadyReq(type='READY', cname='client1', state=State.ON, config=configuration))

## send logs file about tensorboard
def get_file_chunks(filename):
	CHUNK_SIZE = 100*1024*1024
	with open(filename, 'rb') as f:
		while True:
			piece = f.read(CHUNK_SIZE)
			if len(piece) == 0:
				return 
			yield transportRequest(update_req=UpdateReq(type="L", buffer_chunk=piece, title=filename, file_len=len(piece)))

def send_logs(stub, in_file_name):
	chunks_generator = get_file_chunks(in_file_name)
	logs_response = stub.transport(chunks_generator)
	for lr in logs_response:
		print(f"Finish deliver file: {lr.update_rep.title}, type: {lr.update_rep.type}")

##
## send training state to server
def request_training(nclient):
	training_request = [UpdateReq(type="T", cname=nclient, state=State.TRAINING)]
	for tr in training_request:
		yield transportRequest(update_req=tr)

def request_traindone(nclient, cr, bc, acc, loss, tloss, tt, ds, cs, cp, rp, mcp):
	class_size = ','.join(str(e) for e in cs)
	data_size = '';
	for i, d in enumerate(ds):
		if i+1 == len(ds):
			data_size += f'{d}-{ds[d]}'
		else:
			data_size += f'{d}-{ds[d]},'
	
	USTIME = time.time() # upload time
	traindone_request = [UpdateReq(type="D", buffer_chunk=pickle.dumps(bc), state=State.TRAIN_DONE, cname=nclient, current_round=cr, accuracy=acc, loss=loss, tloss=tloss, trainingtime=tt, classsize=class_size, datasize=data_size, uploadtime=str(USTIME), percent_cpu=cp, percent_ram=rp, wrong_pred=mcp)]
	for tr in traindone_request:
		yield transportRequest(update_req=tr)

def request_model_version(mv, cr):
	global client_name

	configuration = dict()
	configuration['model_version'] = Scalar(scint32=mv)
	configuration['current_round'] = Scalar(scint32=cr)
	configuration['client_name'] = Scalar(scstring=client_name)
	version_request = [VersionReq(type="P", config=configuration)]
	for vr in version_request:
		yield transportRequest(version_req=vr)

def send_message(stub):
	global client_name; global acc; global loss; global tloss; global FLAG; global cpu_percent; global ram_percent; global class_for_learning; global mcp

	ready_state = False
	# ready client
	print("### Ready Client ###")
	ready_info_dict = dict()
	ready = request_ready()
	response_ready = stub.transport(ready)
	for rs in response_ready:
		if rs.ready_rep.config['state'].scstring == "SW":
			ready_state = True
			ready_info_dict['cr'] = rs.ready_rep.config['current_round'].scint32
			ready_info_dict['mtr'] = rs.ready_rep.config['max_train_round'].scint32
#ready_info_dict['ga'] = rs.ready_rep.config['model_acc'].scfloat
#ready_info_dict['gl'] = rs.ready_rep.config['model_loss'].scfloat
			ready_info_dict['tmt'] = rs.ready_rep.config['model_type'].scstring
			ready_info_dict['mv'] = rs.ready_rep.config['model_version'].scint32
		else:
			ready_info_dict['cr'] = rs.ready_rep.config['current_round'].scint32
			ready_info_dict['mtr'] = rs.ready_rep.config['max_train_round'].scint32
			ready_info_dict['tmt'] = rs.ready_rep.config['model_type'].scstring
			ready_info_dict['mv'] = rs.ready_rep.config['model_version'].scint32

	if ready_state:
		# update client
		print("### Request Global Model Parameter ###")
		update = request_parameter()
		response_update = stub.transport(update)
		for ru in response_update:
			class_for_learning = learning_fit(ready_info_dict['tmt'], 10, 16, ru.update_rep.buffer_chunk, ready_info_dict['cr'])

		# train client
		training = request_training(client_name)
		response_training = stub.transport(training)

		th1 = Thread(target=cpu_ram_monitoring)
		th1.start()
		acc, loss, tloss, training_time, ds, cs, mcp = class_for_learning.manage_train(cr=ready_info_dict['cr'], cn=client_name) # model fit
		FLAG = False
		time.sleep(10)
		th1.join()
		'''
		# update complete
		## send logs file to server
		print("### Upload model training log files ###")
		for root, dirs, files in os.walk('./send_logs/logs'):
			for fname in files:
				full_fname = os.path.join(root, fname)
				send_logs(stub, full_fname)
		'''
		
		get_params = list()
		while ready_info_dict['cr'] <= ready_info_dict['mtr']:
			print('''#################################################
					 #
					 #
					 #
					 #''')
			print(f"CURRENT ROUND {ready_info_dict['cr']}")
			print('''#
					 #
					 #
					 #
					 #################################################''')
		
			# train done
			print("### Deliver model state: TRAIN DONE to server ###")
			with open('./saved_weight/weights7.pickle', 'rb') as fr:
				get_params = pickle.load(fr)
			traindone = request_traindone(client_name, ready_info_dict['cr'], get_params, acc, loss, tloss, training_time, ds, cs, cpu_percent, ram_percent, mcp)
			response_traindone = stub.transport(traindone)										## 나 학습 다했어!

			oneres_traindone = None; oneres_newround = None
			for rt in response_traindone:
				oneres_traindone = rt

			if oneres_traindone.update_rep.state == State.DELETE:
				print("@@@ Received State.DELETE... finish training... @@@")
				sys.exit(0)

			print(f"### Received from state {oneres_traindone.update_rep.config['state'].scstring} ###")
			# case 1: still learning other model -> state: RESP_ACY
			if oneres_traindone.update_rep.config['state'].scstring == 'RESP_ACY':				## 아직 다른거 학습중이야 기다려
				change_model_version = False
				while True:																		## 응...
					if change_model_version:
						break
					time.sleep(25)														## 30초만 자야지
		
					# check model version
					version = request_model_version(ready_info_dict['mv'], ready_info_dict['cr'])	## 끝났어?
					response_version = stub.transport(version)
					for rv in response_version:
						oneres_newround = rv
					if oneres_newround.version_rep.state == State.NOT_WAIT:						## 어 끝났어!
						ready_info_dict['cr'] = oneres_newround.version_rep.config['current_round'].scint32
						ready_info_dict['mv'] = oneres_newround.version_rep.config['model_version'].scint32
						change_model_version = True
					elif oneres_newround.version_rep.state == State.FIN:
						# 학습이 아예 끝남
						print("all training finish")
						sys.exit(0)
						
				if oneres_newround.version_rep.state == State.DELETE:
					print("@@@ Received State.DELETE... finish training... @@@")
					sys.exit(0)
																								## 아니 안끝났어 더 기다려!
				# train next round
				FLAG = True
				th1 = Thread(target=cpu_ram_monitoring)
				th1.start()
				acc, loss, tloss, training_time, ds, cs, mcp = class_for_learning.manage_train(params=oneres_newround.version_rep.buffer_chunk, cr=ready_info_dict['cr'], cn=client_name)	 ## 다음 라운드 학습해야지
				FLAG = False
				time.sleep(10)
				th1.join()
			# case 2: finish learning one round -> state: RESP_ARY
			elif oneres_traindone.update_rep.config['state'].scstring == 'RESP_ARY':			## 바로 다음 라운드 학습해~
				# train client
				training = request_training(client_name)
				response_training = stub.transport(training)									## 나 학습 시작한다~
				
				ready_info_dict['cr'] = oneres_traindone.update_rep.config['current_round'].scint32
				ready_info_dict['mv'] = oneres_traindone.update_rep.config['model_version'].scint32

				FLAG = True
				th1 = Thread(target=cpu_ram_monitoring)
				th1.start()
				acc, loss, tloss, training_time, ds, cs, mcp = class_for_learning.manage_train(params=oneres_traindone.update_rep.buffer_chunk, cr=ready_info_dict['cr'], cn=client_name)	## 다음 라운드 학습!
				FLAG = False
				time.sleep(10)
				th1.join()

				## for root...		## logs파일 보내는 코드 함수화해서 여기에 넣기
			# case 3: finish all round training
			elif oneres_traindone.update_rep.config['state'].scstring == 'FIN':					## 학습 끝났어!
				ready_info_dict['cr'] = oneres_traindone.update_rep.config['current_round'].scint32
				ready_info_dict['mv'] = oneres_traindone.update_rep.config['model_version'].scint32

				print("all training finish")
				#??? 여기부터 구현해야함
	else: # CTW
		####
		# 현재의 CTW는 위의 코드가 반복되는 부분이 많음
		# 해당 부분을 반드시 수정해서 고쳐야함...
		####
		print("### Received CTW -> Model Version Check ###")		
		
		update_intrusion = request_parameter()
		response_update_intrusion = stub.transport(update_intrusion)
		for ru in response_update_intrusion:
			class_for_learning = learning_fit(ready_info_dict['tmt'], 10, 16, ru.update_rep.buffer_chunk, ready_info_dict['cr'])
		
		change_model_version_intrusion = False
		while True:
			if change_model_version_intrusion:
				break
			time.sleep(50)

			version_intrusion = request_model_version(ready_info_dict['mv'], ready_info_dict['cr'])
			response_version_intrusion = stub.transport(version_intrusion)
			for rvi in response_version_intrusion:
				oneres_newround_intrusion = rvi
			if oneres_newround_intrusion.version_rep.state == State.DELETE:
				change_model_version_intrusion = True
			elif oneres_newround_intrusion.version_rep.state == State.NOT_WAIT:						## 어 끝났어!
				ready_info_dict['cr'] = oneres_newround_intrusion.version_rep.config['current_round'].scint32
				ready_info_dict['mv'] = oneres_newround_intrusion.version_rep.config['model_version'].scint32
				change_model_version_intrusion = True
			elif oneres_newround_intrusion.version_rep.state == State.FIN:
				# 학습이 아예 끝남
				print("all training finish")
				sys.exit(0)
				
		if oneres_newround_intrusion.version_rep.state == State.DELETE:
			print("@@@ Received State.DELETE... finish training... @@@")
			sys.exit(0)

		FLAG = True
		th1 = Thread(target=cpu_ram_monitoring)
		th1.start()
		acc, loss, tloss, training_time, ds, cs, mcp = class_for_learning.manage_train(params=oneres_newround_intrusion.version_rep.buffer_chunk, cr=ready_info_dict['cr'], cn=client_name, state='CTW')	 ## 다음 라운드 학습해야지
		FLAG = False
		time.sleep(10)
		th1.join()
		####
		## ---- 여기서부터 반복되는 내용이 심해짐
		####
		get_params_intrusion = list()
		while ready_info_dict['cr'] <= ready_info_dict['mtr']:
			print('''#################################################
					 #
					 #
					 #
					 #''')
			print(f"CURRENT ROUND {ready_info_dict['cr']}")
			print('''#
					 #
					 #
					 #
					 #################################################''')

			# train done
			print("### Deliver model state: TRAIN DONE to server ###")
			with open('./saved_weight/weights1.pickle', 'rb') as fr:
				get_params_intrusion = pickle.load(fr)
			traindone_intrusion = request_traindone(client_name, ready_info_dict['cr'], get_params_intrusion, acc, loss, tloss, training_time, ds, cs, cpu_percent, ram_percent, mcp)
			response_traindone_intrusion = stub.transport(traindone_intrusion)										## 나 학습 다했어!

			oneres_traindone = None; oneres_newround = None
			for rt in response_traindone_intrusion:
				oneres_traindone = rt

			if oneres_traindone.update_rep.state == State.DELETE:
				print("@@@ Received State.DELETE... finish training... @@@")
				sys.exit(0)

			print(f"### Received from state {oneres_traindone.update_rep.config['state'].scstring} ###")
			# case 1: still learning other model -> state: RESP_ACY
			if oneres_traindone.update_rep.config['state'].scstring == 'RESP_ACY':				## 아직 다른거 학습중이야 기다려
				change_model_version = False
				while True:																		## 응...
					if change_model_version:
						break
					time.sleep(25)														## 30초만 자야지
		
					# check model version
					version = request_model_version(ready_info_dict['mv'], ready_info_dict['cr'])	## 끝났어?
					response_version = stub.transport(version)
					for rv in response_version:
						oneres_newround = rv
					if oneres_newround.version_rep.state == State.DELETE:
						change_model_version = True
					elif oneres_newround.version_rep.state == State.NOT_WAIT:						## 어 끝났어!
						ready_info_dict['cr'] = oneres_newround.version_rep.config['current_round'].scint32
						ready_info_dict['mv'] = oneres_newround.version_rep.config['model_version'].scint32
						change_model_version = True
					elif oneres_newround.version_rep.state == State.FIN:
						# 학습이 아예 끝남
						print("all training finish")
						sys.exit(0)
						
				if oneres_newround_intrusion.version_rep.state == State.DELETE:
					print("@@@ Received State.DELETE... finish training... @@@")
					sys.exit(0)
																								## 아니 안끝났어 더 기다려!
				# train next round
				FLAG = True
				th1 = Thread(target=cpu_ram_monitoring)
				th1.start()
				acc, loss, tloss, training_time, ds, cs, mcp = class_for_learning.manage_train(params=oneres_newround_intrusion.version_rep.buffer_chunk, cr=ready_info_dict['cr'], cn=client_name)	 ## 다음 라운드 학습해야지
				FLAG = False
				time.sleep(10)
				th1.join()
			# case 2: finish learning one round -> state: RESP_ARY
			elif oneres_traindone.update_rep.config['state'].scstring == 'RESP_ARY':			## 바로 다음 라운드 학습해~
				# train client
				training = request_training(client_name)
				response_training = stub.transport(training)									## 나 학습 시작한다~
				
				ready_info_dict['cr'] = oneres_traindone.update_rep.config['current_round'].scint32
				ready_info_dict['mv'] = oneres_traindone.update_rep.config['model_version'].scint32

				FLAG = True
				th1 = Thread(target=cpu_ram_monitoring)
				th1.start()
				acc, loss, tloss, training_time, ds, cs, mcp = class_for_learning.manage_train(params=oneres_traindone.update_rep.buffer_chunk, cr=ready_info_dict['cr'], cn=client_name)	## 다음 라운드 학습!
				FLAG = False
				time.sleep(10)
				th1.join()

				## for root...		## logs파일 보내는 코드 함수화해서 여기에 넣기
			# case 3: finish all round training
			elif oneres_traindone.update_rep.config['state'].scstring == 'FIN':					## 학습 끝났어!
				ready_info_dict['cr'] = oneres_traindone.update_rep.config['current_round'].scint32
				ready_info_dict['mv'] = oneres_traindone.update_rep.config['model_version'].scint32

				print("all training finish")
				# ??? 여기부터 구현해야함

def cpu_ram_monitoring():
	global cpu_percent; global ram_percent

	cpu_list = list(); ram_list = list()
	while FLAG:
		time.sleep(4)
		cpu_list.append(psutil.cpu_percent())
		ram_list.append(psutil.virtual_memory()[2])

	cpu_percent = max(cpu_list)
	ram_percent = max(ram_list)

def run():
	options = [('grpc.max_receive_message_length', 512*1024*1024), ('grcp.max_send_message_length', 512*1024*1024)]
	channel = grpc.insecure_channel('0.0.0.0:8890', options=options)
	stub = TransportServiceStub(channel)
	send_message(stub)

if __name__ == '__main__':
	FLAG = True; ret_value = None; cpu_percent = 0.0; ram_percent = 0.0
	client_name = ""; acc = 0.0; loss = 0.0; tloss = 0.0; mcp = None
	run()
