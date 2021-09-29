from concurrent import futures

import os
import time
import pickle
import sqlite3
import threading
import requests
import numpy as np

import model_evaluate_for_mnist

import grpc
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.transport_pb2 import Scalar, transportResponse, ReadyRep, UpdateRep, VersionRep, State
from common.transport_pb2_grpc import TransportServiceServicer, add_TransportServiceServicer_to_server

readyClientSids = list(); currentRoundClientUpdates = list(); received_parameters = list()
clientUpdateAmount = 0
ALL_CURRENT_ROUND = 0
MODEL_VERSION = 1
MAX_NUM_ROUND = 50
NUM_CLIENTS_CONTACTED_PER_ROUND = 0
CHECK_CLIENT_TRAINING = False; CHECK_TRAIN_TIMER = False

# sqlite3 database
conn_index = sqlite3.connect("./dashboard_db/index.db", check_same_thread=False)
cur_index = conn_index.cursor()
conn_learning = sqlite3.connect("./dashboard_db/learning.db", check_same_thread=False)
cur_learning = conn_learning.cursor()

#################

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

## timer
def client_training_check_timer():
	global CHECK_CLIENT_TRAINING
	time.sleep(30)
	CHECK_CLIENT_TRAINING = True

def ready_client(name, config):
	global ALL_CURRENT_ROUND; global NUM_CLIENTS_CONTACTED_PER_ROUND
	global CHECK_CLIENT_TRAINING; global CHECK_TRAIN_TIMER

	if not CHECK_TRAIN_TIMER:
		CHECK_TRAIN_TIMER = True
		t = threading.Thread(target=client_training_check_timer)
		t.start()

	configuration = dict()
	if not CHECK_CLIENT_TRAINING:
		time.sleep(5)
		if name not in readyClientSids:
			readyClientSids.append(name)
			NUM_CLIENTS_CONTACTED_PER_ROUND += 1
	
		if name in readyClientSids and config['current_round'].scint32 == 0:
			print("### Check Train Round ###")
			current_round = config['current_round'].scint32

			#model_loss, model_acc = trainNextRound(current_round)

			ALL_CURRENT_ROUND = current_round + 1

			configuration['state'] = Scalar(scstring="SW")
			configuration['model_version'] = Scalar(scint32=MODEL_VERSION)
			configuration['current_round'] = Scalar(scint32=ALL_CURRENT_ROUND)
#configuration['model_acc'] = Scalar(scfloat=model_acc)
#configuration['model_loss'] = Scalar(scfloat=model_loss)
			configuration['max_train_round'] = Scalar(scint32=MAX_NUM_ROUND)
			configuration['model_type'] = Scalar(scstring="mobilenet_v2")
	
		return configuration
	else:
		# can't participate in training
		configuration['state'] = Scalar(scstring="CTW")
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
def updateWeight(round_client):
	update_stime = time.time()
	averaged_weight = list()
	for wcl in received_parameters:
		if len(averaged_weight) == 0:
			averaged_weight = wcl
		else:
			for i, wc in enumerate(wcl):
				averaged_weight[i] = averaged_weight[i] + wc

	for i, aw in enumerate(averaged_weight):
		averaged_weight[i] = aw / round_client

	with open('./server_weights/weights.pickle', 'wb') as fw:
		pickle.dump(averaged_weight, fw)
	update_time = time.time() - update_stime
	return update_time

def manage_rounds(nclient, current_round, buffer_chunk, clid):
	global ALL_CURRENT_ROUND; global MODEL_VERSION; global clientUpdateAmount
	global currentRoundClientUpdates; global received_parameters

	print(f"ALL CURRENT ROUND: {ALL_CURRENT_ROUND}, received current round: {current_round}")
	if ALL_CURRENT_ROUND == current_round:
		clientUpdateAmount += 1
		currentRoundClientUpdates.append(nclient)
		received_parameters.append(buffer_chunk)
		if clientUpdateAmount >= NUM_CLIENTS_CONTACTED_PER_ROUND:
			aggre_time = updateWeight(len(currentRoundClientUpdates))
			cur_learning.execute('''SELECT EXISTS (SELECT * FROM ServerTime WHERE round=? and clientid=?)''', (ALL_CURRENT_ROUND, clid,))
			if cur_learning.fetchone()[0]:
				cur_learning.execute('''UPDATE ServerTime SET aggregationtime=? WHERE round=? and clientid=?''', (aggre_time, ALL_CURRENT_ROUND, clid,))
			else:
				cur_learning.execute('''INSERT INTO ServerTime VALUES (?, ?, ?, ?);''', (ALL_CURRENT_ROUND, clid, aggre_time, 0,))
			conn_learning.commit()

			if current_round >= MAX_NUM_ROUND:
				print('All rounds of learning have been completed, and the status is “FIN”.')
				### stop_and_eval() => 아직 구현은 안함... 서버에서 검증하는 코드가 필요한지?
				ALL_CURRENT_ROUND += 1
				MODEL_VERSION += 1
				
				return "FIN"
			else:
				print('The first round of learning has been completed, and the current status is "RESP_ARY".')
				clientUpdateAmount = 0
				currentRoundClientUpdates.clear()
				ALL_CURRENT_ROUND += 1
				MODEL_VERSION += 1

				return "RESP_ARY"

		print('There are clients still learning in the current round, the current status is "RESP_ACY".')
		return "RESP_ACY"


def version_check(Mversion, Cround):
	configuration = dict()
	if MODEL_VERSION == Mversion: # not finish other client training
		return [State.WAIT, configuration]
	elif MODEL_VERSION != Mversion and MAX_NUM_ROUND == Cround: # finish all round traning
		configuration['current_round'] = Scalar(scint32=ALL_CURRENT_ROUND)
		configuration['model_version'] = Scalar(scint32=MODEL_VERSION)
		return ["FIN", configuration]
	elif MODEL_VERSION != Mversion: #finish one round training
		configuration['current_round'] = Scalar(scint32=ALL_CURRENT_ROUND)
		configuration['model_version'] = Scalar(scint32=MODEL_VERSION)
		return [State.NOT_WAIT, configuration]
##

def change_clientname2index(cn):
	cur_index.execute('''SELECT id FROM ClientID WHERE clientname=?''', (cn,))
	return cur_index.fetchone()[0]

def manage_request(request):
	for req in request:
		if req.ready_req.type == 'R':
			# insert db client database
			cur_index.execute('''SELECT COUNT(*) FROM ClientID;''')
			cCount = cur_index.fetchone()[0]
			cur_index.execute('''INSERT INTO ClientID VALUES (?, ?);''', (cCount+1, req.ready_req.cname,))
			cur_learning.execute('''SELECT COUNT(*) FROM NowStatus WHERE round=?;''', (req.ready_req.config['current_round'].scint32,))
			if cur_learning.fetchone()[0] != 0:
				cur_learning.execute('''SELECT status_on FROM NowStatus WHERE round=?;''', (req.ready_req.config['current_round'].scint32,))
				Ston = cur_learning.fetchone()[0]
				cur_learning.execute('''UPDATE NowStatus SET status_on=? WHERE round=?;''', (Ston+1, req.ready_req.config['current_round'].scint32,))
			else:
				cur_learning.execute('''INSERT INTO NowStatus (round, status_on, status_off) VALUES (?, ?, ?);''', (req.ready_req.config['current_round'].scint32, cCount, 0))

			conn_index.commit()

			# api
			#r = requests.get('http://192.168.1.119:5005/new_client')
		
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
			res_normal = [UpdateRep(type=req.update_req.type, title=req.update_req.title, config=dict())]
			for rn in res_normal:
				yield transportResponse(update_rep=rn)
		elif req.update_req.type == 'D':																## 학습이 끝났는데 뭐해야해?
			configuration = dict()
			client_id = change_clientname2index(req.update_req.cname)

			# preprocess datasize, classsize
			classSize = req.update_req.classsize.split(',')
			dataSize = [0 for i in range(len(classSize))]
			for ds in req.update_req.datasize.split(','):
				dataSize[int(ds.split('-')[0])] += int(ds.split('-')[1])

			# save database
			cur_learning.execute('''INSERT INTO LearningTrain VALUES (?, ?, ?, ?, ?);''', (req.update_req.current_round, client_id, req.update_req.accuracy, req.update_req.loss, req.update_req.trainingtime,))
			cur_learning.execute('''INSERT INTO LearningRound VALUES (?, ?, ?, ?);''', (req.update_req.current_round, client_id, ','.join(str(d) for d in dataSize), req.update_req.classsize,))
			cur_learning.execute('''INSERT INTO LearningTime VALUES (?, ?, ?, ?);''', (req.update_req.current_round, client_id, float(req.update_req.uploadtime), time.time(),))
			conn_learning.commit()

			# api
			#r = requests.get('http://192.168.1.119:5005/train_done')

			print("### Start rounds management ###")
			rounds_state = manage_rounds(req.update_req.cname, req.update_req.current_round, pickle.loads(req.update_req.buffer_chunk), client_id)

			configuration['model_version'] = Scalar(scint32=MODEL_VERSION)
			configuration['current_round'] = Scalar(scint32=ALL_CURRENT_ROUND)
			configuration['state'] = Scalar(scstring=rounds_state)
			print(f"### Current state: {rounds_state} ###")
			if rounds_state == "RESP_ACY": # still learning model										## 딴거 학습중이니까 좀있다가 다시 물어봐
				res_rounds = [UpdateRep(type=req.update_req.type, config=configuration)]
				for rr in res_rounds:
					yield transportResponse(update_rep=rr)
			elif rounds_state == "RESP_ARY":															## 학습 다 끝났네. 바로 다음 라운드 학습해
				dis_stime = time.time()
				res_rounds = [UpdateRep(type=req.update_req.type, buffer_chunk=send_parameter(), config=configuration)]
				for rr in res_rounds:
					yield transportResponse(update_rep=rr)
			elif rounds_state == "FIN":																	## 학습 아예 끝났어!
				res_rounds = [UpdateRep(type=req.update_req.type, config=configuration)]
				for rr in res_rounds:
					yield transportResponse(update_rep=rr)
			
			if rounds_state == "RESP_ARY":
				dis_time = time.time() - dis_stime
				cur_learning.execute('''SELECT EXISTS (SELECT * FROM ServerTime WHERE round=? and clientid=?)''', (ALL_CURRENT_ROUND-1, client_id,))
				if cur_learning.fetchone()[0]:
					cur_learning.execute('''UPDATE ServerTime SET distributiontime=? WHERE round=? and clientid=?''', (dis_time, ALL_CURRENT_ROUND-1, client_id,))
				else:
					cur_learning.execute('''INSERT INTO ServerTime VALUES (?, ?, ?, ?);''', (ALL_CURRENT_ROUND-1, client_id, 0, dis_time,))
				conn_learning.commit()
		elif req.version_req.type == 'P':
			now_state = version_check(req.version_req.config['model_version'].scint32, req.version_req.config['current_round'].scint32)
			if now_state[0] == State.NOT_WAIT:
				for ns in [now_state]:
					yield transportResponse(version_rep=VersionRep(state=ns[0], buffer_chunk=send_parameter(), config=ns[1]))
			elif now_state[0] == "FIN":
				for ns in [now_state]:
					yield transportResponse(version_rep=VersionRep(state=ns[0], config=ns[1]))
			else: ## "WAIT"
				for ns in [now_state]:
					yield transportResponse(version_rep=VersionRep(state=ns[0], config=ns[1]))

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
	cur_learning.execute('''INSERT INTO LearningInfo (max_round) VALUES (?);''', (MAX_NUM_ROUND,))

	eval_model = model_evaluate_for_mnist.evaluate_LocalModel(16, 48, np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))
	made_model = eval_model.buildGlobalModel(3, 0.001)
	serve()
