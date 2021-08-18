from concurrent import futures

import os
import time
import grpc
from transport_pb2 import Scalar, transportResponse, ReadyRep, UpdateRep
from transport_pb2_grpc import TransportServiceServicer, add_TransportServiceServicer_to_server
import numpy as np

import model_evaluate

readyClientSids = list(); currentRoundClientUpdates = list(); received_parameters = list()
clientUpdateAmount = 0
ALL_CURRENT_ROUND = 0
MODEL_VERSION = 1
MAX_TRAIN_ROUND = 3
NUM_CLIENTS_CONTACTED_PER_ROUND = 3

## update_req.type == P
def send_parameter():
	return eval_model.get_weights(made_model)

## ready_req.type == READY
def trainNextRound(current_round):
	print("### Round " + str(current_round) + " ###")

	# eval
	model_evalResult = eval_model.train_model_tosave(made_model)
	print("Evaluate Loss : " + str(model_evalResult['loss']) + " Evaluate Accuracy : " + str(model_evalResult['accuracy']))

	# save
	#eval_model.saved_model(made_model) # 모델 저장은 필요한가?

	return model_evalResult['loss'], model_evalResult['accuracy']

def ready_client(name, config):
	configuration = dict()

	if name not in readyClientSids:
		readyClientSids.append(name)

	if name in readyClientSids and config['current_round'].scint32 == 0:
		print("### Check Train Round ###")
		model_round = config['current_round'].scint32
		model_loss, model_acc = trainNextRound(model_round)
		ALL_CURRENT_ROUND = model_round + 1

		configuration['model_version'] = Scalar(scint32=MODEL_VERSION)
		configuration['current_round'] = Scalar(scint32=ALL_CURRENT_ROUND)
		configuration['model_acc'] = Scalar(scfloat=model_acc)
		configuration['model_loss'] = Scalar(scfloat=model_loss)
		configuration['max_train_round'] = Scalar(scint32=MAX_TRAIN_ROUND)
		configuration['model_type'] = Scalar(scstring="mobilenet_v2")

	return configuration

## save logs file from client
def save_chunks_to_file(buffer_chunk, title):
	if not os.path.exists('.'+'/'.join(title[11:].split('/')[:-1])):
		os.makedirs('.'+'/'.join(title[11:].split('/')[:-1]))
	with open('.'+title[11:], 'wb') as fw:
		fw.write(buffer_chunk)
	return True
##
## manage rounds and model version check
def manage_rounds(nclient, current_round, buffer_chunk):
	if ALL_CURRENT_ROUND == current_round:
		clientUpdateAmount += 1
		currentRoundClientUpdates.append(nclient)
		if clientUpdateAmount >= NUM_CLIENTS_CONTACTED_PER_ROUND and len(currentRoundClientUpdates) > 0:
			received_parameters.append(buffer_chunk)
			if current_round >= MAX_NUM_ROUNDS:
			
			else:
				clientUpdateAmount -= 1

		return "RESP_ACY"


def version_check(Mversion, Cround):
	configuration = dict()
	if MODEL_VERSION == Mversion: # not finish other client training
		return [State.WAIT, configuration]
	elif MODEL_VERSION != Mversion and MAX_TRAIN_ROUND == Cround: # finish all round traning
		configuration['current_round'] = Scalar(scint32=ALL_CURRENT_ROUND)
		configuration['model_version'] = Scalar(scint32=MODEL_VERSION)
		return ["FIN", configuration]
	elif MODEL_VERSION != Mversion: #finish one round training
		configuration['current_round'] = Scalar(scint32=ALL_CURRENT_ROUND)
		configuration['model_version'] = Scalar(scint32=MODEL_VERSION)
		return [State.NOT_WAIT, configuration]
##

def manage_request(request):
	for req in request:
		if req.ready_req.type == 'R':
			res_config = [ready_client(req.ready_req.cname, req.ready_req.config)]
			for rc in res_config:
				yield transportResponse(ready_rep=ReadyRep(config=rc))
		elif req.update_req.type == 'P':
			res_para = [send_parameter()]
			for rp in res_para:
				yield transportResponse(update_rep=UpdateRep(type=req.update_req.type, buffer_chunk=rp, title="parameters"))
		elif req.update_req.type == 'T':
			# 바로 아래 elif문에서 같은 내용의 코드 존재
			client_name = req.update_req.cname
			state = req.update_req.state

			res_normal = [UpdateRep(type=req.update_req.type)]
			for rn in res_normal:
				yield transportResponse(update_rep=rn)
		elif req.update_req.type == 'L':
			print(req.update_req.title)
			save_check = save_chunks_to_file(req.update_req.buffer_chunk, req.update_req.title)
			res_normal = [UpdateRep(type=req.update_req.type)]
			for rn in res_normal:
				yield transportResponse(update_rep=rn)
		elif req.update_req.type == 'D':
			rounds_state = manage_rounds(req.update_req.cname, req.update_req.current_round, req.update_req.buffer_chunk)
			if rounds_state == "RESP_ACY": # still learning model
				configuration = dict()
				configuration['model_version'] = Scalar(scint32=MODEL_VERSION)
				configuration['current_round'] = Scalar(scint32=ALL_CURRENT_ROUND)
				configuration['state'] = Scalar(scstring="RESP_ACY")

				res_rounds = [UpdateRep(type=req.update_req.type, config=configuration)]
				for rr in res_rounds:
					yield transportResponse(update_rep=rr)
		elif req.version_req.type == 'P':
			now_state = version_check(req.version_req.model_version, req.version_req.current_round)
			if now_state[0] == State.NOT_WAIT:
				res_config = [now_state]
				for rc in res_config:
					yield transportResponse(version_rep=VersionRep(state=rc[0], buffer_chunk=send_parameter(), config=rc[1]))
			elif now_state[0] == "FIN":

			else:
				for ns in [now_state]:
					yield transportResponse(version_rep=VersionRep(state=ns[0], config=ns[1])

class TransportService(TransportServiceServicer):
	def transport(self, request, context):
		trans_res = manage_request(request)
		return trans_res

def serve():
	options = [('grpc.max_receive_message_length', 512*1024*1024), ('grcp.max_send_message_length', 512*1024*1024)]
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
	add_TransportServiceServicer_to_server(TransportService(), server)
	server.add_insecure_port('[::]:8890')
	server.start()
#server.wait_for_termination()

	try:
		while True:
			time.sleep(60*60*24)
	except KeyboardInterrupt:
		server.stop(0)


if __name__ == '__main__':
	eval_model = model_evaluate.evaluate_LocalModel(16, 224, np.array(['0', '1', '2', '3', '4']))
	made_model = eval_model.buildGlobalModel(3, 0.001)
	serve()
