from __future__ import print_function

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
	ready_request = [ReadyReq(type='READY', cname=client_name, state=State.ON, config=configuration)]
	for rr in ready_request:
		yield transportRequest(ready_req=rr)
	#return transportRequest(ready_req=ReadyReq(type='READY', cname='client1', state=State.ON, config=configuration))

def send_message(stub):
	global client_name

	# ready client
	ready_info_dict = dict()
	ready = request_ready()
	response_ready = stub.transport(ready)
	for rs in response_ready:
		ready_info_dict['cr'] = rs.ready_rep.coonfig['current_round']
		ready_info_dict['mtr'] = rs.ready_rep.coonfig['max_train_round']
		ready_info_dict['ga'] = rs.ready_rep.coonfig['model_acc']
		ready_info_dict['gl'] = rs.ready_rep.coonfig['model_loss']
		ready_info_dict['tmt'] = rs.ready_rep.coonfig['model_type']
		ready_info_dict['mv'] = rs.ready_rep.coonfig['model_version']

	# update client
	update = request_parameter(ready_info_dict)
	response_update = stub.transport(update)
	for ru in response_update:
		class_for_learning = learning_fit(ready_info_dict['mv'], 16, ru.update_rep.buffer_chunk, ready_info_dict['cr'])
	print(f"### Model Training - Round: {ready_info_dict['cr']} ###")
	get_params = class_for_learning.manage_train() # model fit
	# update complete
	

def run():
	channel = grpc.insecure_channel('localhost:8890')
	stub = TransportServiceStub(channel)
	send_message(stub)

if __name__ == '__main__':
	client_name = ""
	run()
