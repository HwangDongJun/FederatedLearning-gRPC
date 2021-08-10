from __future__ import print_function

import grpc
from transport_pb2 import Scalar, transportRequest, ReadyReq, UpdateReq, State
from transport_pb2_grpc import TransportServiceStub


def request_update():
	update_request = [UpdateReq(type="P")]
	for ur in update_request:
		yield transportRequest(update_req=ur)

def request_ready():
	configuration = dict()
	configuration['current_round'] = Scalar(scint32=0)

	ready_request = [ReadyReq(type='READY', cname='client1', state=State.ON, config=configuration)]
	for rr in ready_request:
		yield transportRequest(ready_req=rr)
	#return transportRequest(ready_req=ReadyReq(type='READY', cname='client1', state=State.ON, config=configuration))

def send_message(stub):
	ready_info_dict = dict()
	ready = request_ready()
	response_ready = stub.transport(ready)
	for rs in response_ready:
		ready_info_dict['current_round'] = rs.ready_rep.coonfig['current_round']
		ready_info_dict['max_train_round'] = rs.ready_rep.coonfig['max_train_round']
		ready_info_dict['gmodel_acc'] = rs.ready_rep.coonfig['model_acc']
		ready_info_dict['gmodel_loss'] = rs.ready_rep.coonfig['model_loss']
		ready_info_dict['train_model_type'] = rs.ready_rep.coonfig['model_type']
		ready_info_dict['model_version'] = rs.ready_rep.coonfig['model_version']

	update = request_update(ready_info_dict)
	response_update = stub.transport(update)
	for ru in response_update:
		print(ru)

def run():
	channel = grpc.insecure_channel('localhost:8890')
	stub = TransportServiceStub(channel)
	send_message(stub)

if __name__ == '__main__':
	run()
