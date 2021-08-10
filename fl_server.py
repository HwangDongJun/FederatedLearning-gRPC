from concurrent import futures

import time
import grpc
from transport_pb2 import Scalar, transportResponse, ReadyRep, UpdateRep
from transport_pb2_grpc import TransportServiceServicer, add_TransportServiceServicer_to_server
import numpy as np

import model_evaluate

readyClientSids = list()
ALL_CURRENT_ROUND = 0
MODEL_VERSION = 0
MAX_TRAIN_ROUND = 3

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

def manage_request(request):
	for req in request:
		if req.ready_req.type == 'READY':
			res_config = [ready_client(req.ready_req.cname, req.ready_req.config)]
			for rc in res_config:
				yield transportResponse(ready_rep=ReadyRep(config=rc))
		elif req.update_req.type == 'P':
			res_para = [send_parameter()]
			for rp in res_para:
				yield transportResponse(update_rep=UpdateRep(type=req.update_req.type, buffer_chunk=rp, title="parameters"))

class TransportService(TransportServiceServicer):
	def transport(self, request, context):
		trans_res = manage_request(request)
		return trans_res

def serve():
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
