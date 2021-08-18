from __future__ import print_function

import os
import random
import grpc
from transport_pb2 import Scalar, transportRequest, ReadyReq, UpdateReq, State
from transport_pb2_grpc import TransportServiceStub

from client_fit_model import learning_fit

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
			yield transportRequest(update_req=UpdateReq(type="L", buffer_chunk=piece, title=filename, state=State.TRAIN_DONE, file_len=len(piece)))

def send_logs(stub, in_file_name):
	chunks_generator = get_file_chunks(in_file_name)
	logs_response = stub.transport(chunks_generator)

##
## send training state to server
def request_training(nclient):
	training_request = [UpdateReq(type="T", cname=nclient, state=State.TRAINING)]
	for tr in training_request:
		yield transportRequest(update_req=tr)

def send_message(stub):
	global client_name

	# ready client
	print("### Ready Client ###")
	ready_info_dict = dict()
	ready = request_ready()
	response_ready = stub.transport(ready)
	for rs in response_ready:
		ready_info_dict['cr'] = rs.ready_rep.config['current_round'].scint32
		ready_info_dict['mtr'] = rs.ready_rep.config['max_train_round'].scint32
		ready_info_dict['ga'] = rs.ready_rep.config['model_acc'].scfloat
		ready_info_dict['gl'] = rs.ready_rep.config['model_loss'].scfloat
		ready_info_dict['tmt'] = rs.ready_rep.config['model_type'].scstring
		ready_info_dict['mv'] = rs.ready_rep.config['model_version'].scint32

	# update client
	print("### Request Global Model Parameter ###")
	update = request_parameter()
	response_update = stub.transport(update)
	for ru in response_update:
		class_for_learning = learning_fit(ready_info_dict['tmt'], 1, 16, ru.update_rep.buffer_chunk, ready_info_dict['cr'])

	# train client
	training = request_training(client_name)
	response_training = stub.transport(training)

	print(f"### Model Training - Round: {ready_info_dict['cr']} ###")
	get_params = class_for_learning.manage_train() # model fit

	# update complete
	## send logs file to server
	print("### Upload model training log files ###")
	for root, dirs, files in os.walk('./send_logs/logs'):
		for fname in files:
			full_fname = os.path.join(root, fname)
			send_logs(stub, full_fname)

	# 

def run():
	options = [('grpc.max_receive_message_length', 512*1024*1024), ('grcp.max_send_message_length', 512*1024*1024)]
	channel = grpc.insecure_channel('localhost:8890', options=options)
	stub = TransportServiceStub(channel)
	send_message(stub)

if __name__ == '__main__':
	client_name = ""
	run()
