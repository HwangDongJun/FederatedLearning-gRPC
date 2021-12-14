from concurrent import futures

import os
import time
import json
import random
import pickle
import sqlite3
import threading
import itertools
import requests
import numpy as np

#import model_evaluate_for_mnist
import model_evaluate

import grpc
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from common.transport_pb2 import Scalar, transportResponse, ReadyRep, UpdateRep, VersionRep, State
from common.transport_pb2_grpc import TransportServiceServicer, add_TransportServiceServicer_to_server

lock = threading.Lock()

readyClientSids = list(); currentRoundClientUpdates = list(); received_parameters = list()
intrusionClient = list()
clientUpdateAmount = 0
ALL_CURRENT_ROUND = 0
MODEL_VERSION = 1
MAX_NUM_ROUND = 200
NUM_CLIENTS_CONTACTED_PER_ROUND = 0
CHECK_CLIENT_TRAINING = False; CHECK_TRAIN_TIMER = False

for_loss_select_client = set()
datasize_client = set()
wrong_pred_client = set()

data_q = list()
reverse_bool = False

delete_client_list = list()
time_client_list = list()
s_list = list()
s_time_dict = dict()
tsd = 0
tsdk = 0
tkul = 0
tkud = 0
seta = 0
del_par_cli = 0
acc_pass = False
entropy_pass = False
pred_pass = False
time_pass = False
cs_time = 0
taa = 0
temp_cs_time = 0
total_cs_time = 0
total_start_cs_time = 0

random_count = 0

data_overload_q = list()
check_overload = False
overload_pass = False

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
	print("### Wait Client ###")
	time.sleep(30)
	print("### Time out! Don't received client ###")
	CHECK_CLIENT_TRAINING = True

def ready_client(name, config):
	global ALL_CURRENT_ROUND; global NUM_CLIENTS_CONTACTED_PER_ROUND
	global CHECK_CLIENT_TRAINING; global CHECK_TRAIN_TIMER; global intrusionClient

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
			configuration['max_train_round'] = Scalar(scint32=MAX_NUM_ROUND)
			configuration['model_type'] = Scalar(scstring="mobilenet_v2")
	
		return configuration
	else:
		print("### Dectect client engagement. Client will start learning from the next round. ###")
		# can't participate in training
		configuration['state'] = Scalar(scstring="CTW")
		configuration['model_version'] = Scalar(scint32=MODEL_VERSION)
		configuration['current_round'] = Scalar(scint32=ALL_CURRENT_ROUND)
		configuration['max_train_round'] = Scalar(scint32=MAX_NUM_ROUND)
		configuration['model_type'] = Scalar(scstring="mobilenet_v2")
		
		intrusionClient.append(name)
		print(f"### Current intrusion client list: {len(intrusionClient)} ###")
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
	aggregation_stime = time.time()
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
		
	if ALL_CURRENT_ROUND != 0 and ALL_CURRENT_ROUND % 20 == 0:
		with open(f'./server_weights/weights_{ALL_CURRENT_ROUND}.pickle', 'wb') as fw:
			pickle.dump(averaged_weight, fw)
	aggregation_time = time.time() - aggregation_stime
	return aggregation_time

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
			received_parameters = list()
			try:
				lock.acquire(True)
				cur_learning.execute('''INSERT INTO AggregationTime VALUES (?, ?);''', (ALL_CURRENT_ROUND, aggre_time,))
				conn_learning.commit()
			finally:
				lock.release()

			if current_round >= MAX_NUM_ROUND:
				print('All rounds of learning have been completed, and the status is “FIN”.')
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
	global delete_client_list; global NUM_CLIENTS_CONTACTED_PER_ROUND
	global for_loss_select_client; global datasize_client; global wrong_pred_client
	global time_client_list; global s_list; global tsd; global tsdk; global tkul; global tkud; global seta; global del_par_cli; global time_pass; global acc_pass; global cs_time; global taa; global temp_cs_time; global s_time_list; global entropy_pass; global pred_pass
	global random_count
	global data_q; global reverse_bool
	global total_cs_time; global total_start_cs_time
	global overload_pass; global check_overload; global data_overload_q
	global intrusionClient
	
	for req in request:
		if req.ready_req.type == 'R':
			# insert db client database
			if NUM_CLIENTS_CONTACTED_PER_ROUND != 0:
				cur_index.execute('''SELECT id FROM ClientID ORDER BY id DESC LIMIT 1;''')
				cCount = cur_index.fetchone()[0]
			else:
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
			# client selection
			curr_oneround = req.update_req.current_round
			#if len(delete_client_list) != 0:
			#	time.sleep(10)
			# delete_client_list = list(); total_cs_time = 0; temp_time_del_list = list()
			# if os.path.isfile(f"/home/dnlab/Downloads/check_list_{curr_oneround}.txt"):
			# 	print("Find client selection file!")
			# 	select_json = ""
			# 	with open(f"/home/dnlab/Downloads/check_list_{curr_oneround}.txt") as json_file:
			# 		select_json = json.load(json_file)

			# 	###
			# 	sel_class = ""
			# 	if ',' in select_json['sel_class']:
			# 		sel_class = select_json['sel_class'].split(',')
			# 	else:
			# 		sel_class = select_json['sel_class']
			# 	###

			# 	select_client = set()
			# 	if select_json['loss'] == 1:
			# 		while acc_pass:
			# 			time.sleep(10)
					
			# 		acc_pass = True
			# 		cur_learning.execute('''SELECT SUM(loss)/COUNT(loss) FROM LearningTrain WHERE round=? GROUP BY round''', (curr_oneround-1,))
			# 		avg_loss = cur_learning.fetchone()[0]
			# 		cur_learning.execute('''SELECT clientid FROM LearningTrain WHERE round=? and loss >= ? ORDER BY loss desc''', (curr_oneround-1, avg_loss,))
			# 		for row in cur_learning:
			# 			client_id = row[0]
			# 			for_loss_select_client.add(client_id)
						
			# 		temp_for_loss_select_client = for_loss_select_client
			# 		for dcl in temp_for_loss_select_client:
			# 			cur_learning.execute('''SELECT datasize FROM LearningRound WHERE round=? and clientid=?''', (curr_oneround-1,dcl,))
			# 			data_list = cur_learning.fetchone()[0].split(',')
			# 			for sc in sel_class:
			# 				if data_list[int(sc)] != "0":
			# 					for_loss_select_client.discard(dcl)
			# 		if len(for_loss_select_client) >= 2:
			# 			for_loss_select_client = set(itertools.islice(for_loss_select_client, 2))
					
			# 		acc_pass = False
			# 	if select_json['entropy'] == 1:
			# 		while entropy_pass:
			# 			time.sleep(10)
				
			# 		entropy_pass = True
			# 		class_data_list = select_json['dataclass'].split(',')
			# 		cur_learning.execute('''SELECT clientid, datasize FROM LearningRound WHERE round=?''', (curr_oneround-1,))
			# 		for row in cur_learning:
			# 			client_id = row[0]
			# 			client_data = row[1].split(',')		
			# 			for cdl in class_data_list:
			# 				if client_data[int(cdl)] == "0":
			# 					datasize_client.add(client_id)
								
			# 		temp_datasize_client = datasize_client
			# 		for dcl in temp_datasize_client:
			# 			cur_learning.execute('''SELECT datasize FROM LearningRound WHERE round=? and clientid=?''', (curr_oneround-1,dcl,))
			# 			data_list = cur_learning.fetchone()[0].split(',')
			# 			for sc in sel_class:
			# 				if data_list[int(sc)] != "0":
			# 					datasize_client.discard(dcl)
			# 		if len(datasize_client) >= 2:
			# 			datasize_client = set(itertools.islice(datasize_client, 2))
							
			# 		entropy_pass = False
			# 	if select_json['time'] == 1:
			# 		total_start_cs_time = time.time()
			# 		time_pass = False; total_cs_time = 0; s_list = list()
			# 		tsd = 0; tsdk = 0; tkul = 0; tkud = 0; seta = 0; taa = 0; temp_cs_time = 0
			# 		del_par_cli += 1
					
			# 		data_q.append(req.update_req.cname)
			# 		while len(data_q) != NUM_CLIENTS_CONTACTED_PER_ROUND:
			# 			time.sleep(2)
			# 		#while time_pass:
			# 		#	time.sleep(6)
			# 		#cur_learning.execute('''SELECT distributiontime AS sdis FROM DistributionTime WHERE round=? and clientid=?''', (curr_oneround-1, clid,))
			# 		#tsd += cur_learning.fetchone()[0]
			# 		#cur_learning.execute('''SELECT distributiontime AS supl FROM DistributionTime WHERE round=? and clientid=?''', (curr_oneround-1, clid,))
			# 		#tsdk = tsd + cur_learning.fetchone()[0]
			# 		#cur_learning.execute('''SELECT uploadendtime-uploadstarttime AS sdis FROM LearningTime WHERE round=? and clientid=?''', (curr_oneround-1, clid,))
			# 		#tkul = cur_learning.fetchone()[0]
			# 		#cur_learning.execute('''SELECT trainingtime AS traint FROM LearningTrain WHERE round=? and clientid=?''', (curr_oneround-1, clid,))
			# 		#tkud = cur_learning.fetchone()[0]
					
			# 		#1 / tsd - tsdk + tkul + max([0, tkud-seta])
					
			# 		s_time_dict = dict()
			# 		if not reverse_bool:
			# 			cs_time = time.time()
			# 			reverse_bool = True
			# 			time.sleep(2)
			# 			data_q.reverse()
						
			# 			tround = 2000
			# 			for dq in data_q:
			# 				clid = change_clientname2index(dq)
			# 				cur_learning.execute('''SELECT uploadendtime-uploadstarttime AS sdis FROM LearningTime WHERE round=? and clientid=?''', (curr_oneround-1, clid,))
			# 				tkul = cur_learning.fetchone()[0]
			# 				cur_learning.execute('''SELECT trainingtime AS traint FROM LearningTrain WHERE round=? and clientid=?''', (curr_oneround-1, clid,))
			# 				tkud = cur_learning.fetchone()[0]
			# 				seta_t = seta + tkul + max(0, tkud - seta)
							
			# 				for s in s_list:
			# 					cur_learning.execute('''SELECT distributiontime AS sdis FROM DistributionTime WHERE round=? and clientid=?''', (curr_oneround-1, s,))
			# 					tsd += cur_learning.fetchone()[0]
			# 				cur_learning.execute('''SELECT distributiontime AS sdis FROM DistributionTime WHERE round=? and clientid=?''', (curr_oneround-1, clid,))
			# 				tsdk = tsd + cur_learning.fetchone()[0]
			# 				cur_learning.execute('''SELECT aggregationtime AS sdis FROM AggregationTime WHERE round=?''', (curr_oneround-1,))
			# 				taa = cur_learning.fetchone()[0]
			# 				t = temp_cs_time + tsdk + seta_t + taa
							
			# 				#cur_learning.execute('''SELECT SUM(uploadendtime-uploadstarttime)/COUNT(round) AS sdis FROM LearningTime WHERE round<=? and clientid=?''', (curr_oneround-2, clid,))
			# 				#t1_upload = cur_learning.fetchone()[0]
			# 				#cur_learning.execute('''SELECT SUM(trainingtime)/COUNT(round) AS traint FROM LearningTrain WHERE round<=? and clientid=?''', (curr_oneround-2, clid,))
			# 				#t1_train = cur_learning.fetchone()[0]
			# 				#cur_learning.execute('''SELECT SUM(distributiontime)/COUNT(round) AS sdis FROM DistributionTime WHERE round<=? and clientid=?''', (curr_oneround-2, clid,))
			# 				#t1_dis = cur_learning.fetchone()[0]
			# 				#cur_learning.execute('''SELECT SUM(aggregationtime)/COUNT(round) AS sdis FROM AggregationTime WHERE round<=?''', (curr_oneround-2,))
			# 				#t1_agg = cur_learning.fetchone()[0]
			# 				#tround = t1_upload + t1_train + t1_dis + t1_agg
			# 				#cur_learning.execute('''SELECT max(uploadendtime-uploadstarttime) AS sdis FROM LearningTime WHERE round<=? and clientid=?''', (curr_oneround-2, clid,))
			# 				#t1_upload = cur_learning.fetchone()[0]
			# 				#cur_learning.execute('''SELECT max(trainingtime) AS traint FROM LearningTrain WHERE round<=? and clientid=?''', (curr_oneround-2, clid,))
			# 				#t1_train = cur_learning.fetchone()[0]
			# 				#cur_learning.execute('''SELECT max(distributiontime) AS sdis FROM DistributionTime WHERE round<=? and clientid=?''', (curr_oneround-2, clid,))
			# 				#t1_dis = cur_learning.fetchone()[0]
			# 				#cur_learning.execute('''SELECT max(aggregationtime) AS sdis FROM AggregationTime WHERE round<=?''', (curr_oneround-2,))
			# 				#t1_agg = cur_learning.fetchone()[0]
			# 				#tround = t1_upload + t1_train + t1_dis + t1_agg
							
			# 				print(f"[clid]: {clid} -- [seta]: {seta}, [t]: {t}, [tround]: {tround}, [temp_cs_time]: {temp_cs_time}")
			# 				#print(f"[clid]: {clid} -- [seta]: {seta}, [t]: {t}, [temp_cs_time]: {temp_cs_time}")
			# 				if t < tround:
			# 					seta = seta_t
			# 					s_list.append(clid)
			# 				else:
			# 					s_time_dict[clid] = t
			# 				#seta = seta_t
			# 				#s_list.append(clid)
			# 				#s_time_dict[clid] = t
			# 				temp_cs_time = time.time() - cs_time
			# 			time_pass = True
			# 		else:
			# 			while not time_pass:
			# 				time.sleep(4)
			# 			reverse_bool = False
			# 			data_q = list()
					
			# 		#cur_learning.execute('''SELECT SUM(distributiontime)/COUNT(distributiontime) FROM DistributionTime WHERE round=? GROUP BY round''', (curr_oneround-1,))
			# 		#avg_dis_time = cur_learning.fetchone()[0]
			# 		#cur_learning.execute('''SELECT clientid FROM DistributionTime WHERE round=? and distributiontime > ? GROUP BY round''', (curr_oneround-1, avg_dis_time,))
			# 		#for row in cur_learning:
			# 		#	client_id = row[0]
			# 		#	select_client.add(client_id)
			# 		#cur_learning.execute('''SELECT SUM(trainingtime)/COUNT(trainingtime) FROM LearningTrain WHERE round=? GROUP BY round''', (curr_oneround-1,))
			# 		#avg_train_time = cur_learning.fetchone()[0]
			# 		#cur_learning.execute('''SELECT clientid FROM LearningTrain WHERE round=? and trainingtime > ? GROUP BY round''', (curr_oneround-1, avg_train_time,))
			# 		#for row in cur_learning:
			# 		#	client_id = row[0]
			# 		#	select_client.add(client_id)
			# 		#cur_learning.execute('''SELECT SUM(uploadendtime-uploadstarttime)/COUNT(uploadendtime-uploadstarttime) AS avg_upl FROM LearningTime WHERE round=? GROUP BY round''', (curr_oneround-1,))
			# 		#avg_upl_time = cur_learning.fetchone()[0]
			# 		#cur_learning.execute('''SELECT clientid FROM LearningTime WHERE round=? and (uploadendtime-uploadstarttime) > ? GROUP BY round''', (curr_oneround-1, avg_upl_time,))
			# 		#for row in cur_learning:
			# 		#	client_id = row[0]
			# 		#	select_client.add(client_id)
			# 	if select_json['prediction'] == 1:
			# 		while pred_pass:
			# 			time.sleep(10)
						
			# 		pred_pass = True
			# 		input_pred = select_json['prediction_baseline'].split(',')
			# 		pred_baseline = float(input_pred[0])
			# 		pred_round = int(input_pred[1])
					
			# 		cur_learning.execute('''SELECT clientid FROM Predictions WHERE prediction>=? and round=? ORDER BY prediction DESC''', (pred_baseline, pred_round,))
			# 		for row in cur_learning:
			# 			wrong_pred_client.add(row[0])
						
			# 		if len(wrong_pred_client) >= 2:
			# 			wrong_pred_client = set(itertools.islice(wrong_pred_client, 2))
					
			# 		pred_pass = False
				
			# 	#if select_json['overload'] == 1: # 일단 냅두기
			# 	#	overload_pass = False	
			# 	#	data_overload_q.append(req.update_req.cname)
			# 	#	while len(data_overload_q) != NUM_CLIENTS_CONTACTED_PER_ROUND:
			# 	#		time.sleep(2)
				
			# 	#	time.sleep(2)
			# 	#	if not check_overload:
			# 	#		check_overload = True
			# 	#		temp_client_set = set()
			# 	#		cur_learning.execute('''SELECT clientid FROM CPURAMMonitoring WHERE round=? and cpu > ?''', (curr_oneround-1, int(select_json['overload_baseline'].split(',')[0]),))
			# 	#		for row in cur_learning:
			# 	#			temp_client_set.add(row[0])
							
			# 	#		print(temp_client_set)
							
			# 	#		for tcs in temp_client_set:
			# 	#			clid = tcs
			# 	#			cur_learning.execute('''SELECT uploadendtime-uploadstarttime AS sdis FROM LearningTime WHERE round=? and clientid=?''', (curr_oneround-1, clid,))
			# 	#			tkul = cur_learning.fetchone()[0]
			# 	#			cur_learning.execute('''SELECT trainingtime AS traint FROM LearningTrain WHERE round=? and clientid=?''', (curr_oneround-1, clid,))
			# 	#			tkud = cur_learning.fetchone()[0]
			# 	#			cur_learning.execute('''SELECT distributiontime AS sdis FROM DistributionTime WHERE round=? and clientid=?''', (curr_oneround-1, clid,))
			# 	#			tdis = cur_learning.fetchone()[0]
			# 	#			cur_time = tkul + tkud + tdis
							
			# 	#			cur_learning.execute('''SELECT uploadendtime-uploadstarttime AS sdis FROM LearningTime WHERE round=? and clientid=?''', (curr_oneround-2, clid,))
			# 	#			tkul = cur_learning.fetchone()[0]
			# 	#			cur_learning.execute('''SELECT trainingtime AS traint FROM LearningTrain WHERE round=? and clientid=?''', (curr_oneround-2, clid,))
			# 	#			tkud = cur_learning.fetchone()[0]
			# 	#			cur_learning.execute('''SELECT distributiontime AS sdis FROM DistributionTime WHERE round=? and clientid=?''', (curr_oneround-2, clid,))
			# 	#			tdis = cur_learning.fetchone()[0]
			# 	#			pre_time = tkul + tkud + tdis
						
			# 	#			if cur_time < pre_time:
			# 	#				select_client.add(clid)
							
			# 	#		if len(select_client) > 2:
			# 				#cur_learning.execute('''SELECT clientid FROM CPURAMMonitoring WHERE round=? and cpu > ? ORDER BY cpu desc LIMIT 2''', (curr_oneround-1, select_json['overload_baseline'],))
			# 				#select_client = set()
			# 				#for row in cur_learning:
			# 				#	select_client.add(row[0])
			# 	#			select_client = set(random.sample(select_client, 2))
			# 	#		overload_pass = True
			# 	#	else:
			# 	#		while not overload_pass: 
			# 	#			time.sleep(4)
			# 	#		check_overload = False
			# 	#		data_overload_q = list()
				
			# 	if select_json["multicriteria"] != "":
			# 		time.sleep(random.choice([1, 2, 2, 2, 3, 3, 3, 3, 4]))
			# 		print(f"Accruacy: {for_acc_select_client}")
			# 		print(f"Time: {s_list}")
			# 		print(f"System: {select_client}")
			# 		if select_json["multicriteria"] == "acc,time,":
			# 			cur_index.execute('''SELECT id FROM ClientID''')
			# 			for row in cur_index:
			# 				clid = row[0]
			# 				if clid not in list(set(s_list)):
			# 					temp_time_del_list.append(clid)
			# 			for fas in for_acc_select_client:
			# 				if fas in temp_time_del_list:
			# 					delete_client_list.append(fas)
			# 			if len(delete_client_list) == 0:
			# 				temp_list = list(for_acc_select_client)
			# 				temp_list.extend(temp_time_del_list)
			# 				delete_client_list = random.sample(temp_list, 2)
			# 		elif select_json["multicriteria"] == "acc,sys,":
			# 			for fas in for_acc_select_client:
			# 				if fas in select_client:
			# 					delete_client_list.append(fas)
			# 			if len(delete_client_list) == 0:
			# 				temp_list = list(for_acc_select_client)
			# 				temp_list.extend(list(select_client))
			# 				delete_client_list = random.sample(temp_list, 2)
			# 		elif select_json["multicriteria"] == "time,sys,":
			# 			cur_index.execute('''SELECT id FROM ClientID''')
			# 			for row in cur_index:
			# 				clid = row[0]
			# 				if clid not in list(set(s_list)):
			# 					temp_time_del_list.append(clid)
			# 			for sc in select_client:
			# 				if sc in temp_time_del_list:
			# 					delete_client_list.append(sc)
			# 			if len(delete_client_list) == 0:
			# 				temp_list = list(select_client)
			# 				temp_list.extend(temp_time_del_list)
			# 				delete_client_list = random.sample(temp_list, 2)
			# 		elif select_json["multicriteria"] == "acc,time,sys,":
			# 			cur_index.execute('''SELECT id FROM ClientID''')
			# 			for row in cur_index:
			# 				clid = row[0]
			# 				if clid not in list(set(s_list)):
			# 					temp_time_del_list.append(clid)
			# 			acc_system_list = list(set(for_acc_select_client + select_client))
			# 			for asl in acc_system_list:
			# 				if asl in temp_time_del_list:
			# 					delete_client_list.append(asl)
			# 	else: # multicriteria가 아닌 경우
			# 		if len(select_client) != 0:
			# 			delete_client_list = list(select_client)
			# 		elif len(for_loss_select_client) != 0:
			# 			delete_client_list = list(for_loss_select_client)
			# 			for_loss_select_client = set()
			# 		elif len(datasize_client) != 0:
			# 			delete_client_list = list(datasize_client)
			# 			datasize_client = set()
			# 		elif len(wrong_pred_client) != 0:
			# 			delete_client_list = list(wrong_pred_client)
			# 			wrong_pred_client = set()
			# 		elif len(s_list) != 0:
			# 			#print(s_list)
			# 			cur_index.execute('''SELECT id FROM ClientID''')
			# 			for row in cur_index:
			# 				clid = row[0]
			# 				if clid not in list(set(s_list)):
			# 					delete_client_list.append(clid)
			# 			s_list = list()
						
			# 			if len(delete_client_list) > 2:
			# 				sort_s_time_dict = sorted(s_time_dict.items(), reverse=True, key=lambda item: item[1])
			# 				delete_client_list = list()
			# 				delete_client_list.append(sort_s_time_dict[0][0])
			# 				delete_client_list.append(sort_s_time_dict[1][0])
					
			
			# 	#if random_count <= 10:
			# 	#	delete_client_list = [1, 7, 9, 10]
			# 	#	random_count += 1
			# 	#elif random_count >= 11:
			# 	#	delete_client_list = [2, 3, 6, 8]
			# 	#	random_count += 1
			# 	#print(delete_client_list)
				
			# 	total_cs_time = time.time() - total_start_cs_time

			cur_learning.execute('''SELECT remove_client FROM SelectionClient WHERE round=?''', (curr_oneround,))
			delete_client_list = list(map(int, cur_learning.fetchone()[0].split(',')))

			configuration = dict()
			client_id = change_clientname2index(req.update_req.cname)

			# preprocess datasize, classsize
			classSize = req.update_req.classsize.split(',')
			dataSize = [0 for i in range(len(classSize))]
			for ds in req.update_req.datasize.split(','):
				dataSize[int(ds.split('-')[0])] += int(ds.split('-')[1])

			#print(f"@@@ ENDTIME: {time.time()}, STARTTIME: {float(req.update_req.uploadtime)}, CS_TIME: {total_cs_time}") 
			#print(f"INSERT TO DB: {client_id}")
			try:
				lock.acquire(True)
				# save database
				cur_learning.execute('''INSERT INTO LearningTrain VALUES (?, ?, ?, ?, ?, ?);''', (req.update_req.current_round, client_id, req.update_req.accuracy, req.update_req.loss, req.update_req.tloss, req.update_req.trainingtime,))
				cur_learning.execute('''INSERT INTO LearningRound VALUES (?, ?, ?, ?);''', (req.update_req.current_round, client_id, ','.join(str(d) for d in dataSize), req.update_req.classsize,))
				cur_learning.execute('''INSERT INTO LearningTime VALUES (?, ?, ?, ?);''', (req.update_req.current_round, client_id, float(req.update_req.uploadtime), time.time()-total_cs_time,))
				cur_learning.execute('''INSERT INTO CPURAMMonitoring VALUES (?, ?, ?, ?);''', (req.update_req.current_round, client_id, req.update_req.percent_cpu, req.update_req.percent_ram,))
				cur_learning.execute('''INSERT INTO Predictions VALUES (?, ?, ?);''', (req.update_req.current_round, client_id, req.update_req.wrong_pred,))
				conn_learning.commit()
			finally:
				lock.release()

			#print(f"COMPLETE TO DB: {client_id}")

			# api
			#r = requests.get('http://192.168.1.119:5005/train_done')

			print("### Start rounds management ###")
			rounds_state = manage_rounds(req.update_req.cname, req.update_req.current_round, pickle.loads(req.update_req.buffer_chunk), client_id)

			print(f"### [Update] client_id: {client_id} and delete_client_list: {delete_client_list} ///  {overload_pass}###")
			if int(client_id) in delete_client_list:
				print(f"@@@ Delete client - {client_id} @@@")
				NUM_CLIENTS_CONTACTED_PER_ROUND -= 1
				cur_index.execute('''DELETE FROM ClientID WHERE id=?''', (client_id,))
				conn_index.commit()
				delete_rounds = [UpdateRep(type=req.update_req.type, config=configuration, state=State.DELETE)]
				for dr in delete_rounds:
					yield transportResponse(update_rep=dr)
			else:
				configuration['model_version'] = Scalar(scint32=MODEL_VERSION)
				configuration['current_round'] = Scalar(scint32=ALL_CURRENT_ROUND)
				configuration['state'] = Scalar(scstring=rounds_state)
				print(f"### Current state: {rounds_state} ###")
				if rounds_state == "RESP_ACY": # still learning model										## 딴거 학습중이니까 좀있다가 다시 물어봐
					res_rounds = [UpdateRep(type=req.update_req.type, config=configuration, state=State.ON)]
					for rr in res_rounds:
						yield transportResponse(update_rep=rr)
				elif rounds_state == "RESP_ARY":															## 학습 다 끝났네. 바로 다음 라운드 학습해
					if len(intrusionClient) != 0:
						NUM_CLIENTS_CONTACTED_PER_ROUND += len(intrusionClient)
						intrusionClient = list()
					
					dis_stime = time.time()
					send_params = list()
					with open('./server_weights/weights.pickle', 'rb') as fr:
						send_params = pickle.load(fr)
					res_rounds = [UpdateRep(type=req.update_req.type, buffer_chunk=pickle.dumps(send_params), config=configuration, state=State.ON)]
					for rr in res_rounds:
						yield transportResponse(update_rep=rr)
				elif rounds_state == "FIN":																	## 학습 아예 끝났어!
					res_rounds = [UpdateRep(type=req.update_req.type, config=configuration, state=State.ON)]
					for rr in res_rounds:
						yield transportResponse(update_rep=rr)
				
				if rounds_state == "RESP_ARY":
					os.system('rm -rf /home/dnlab/FederatedLearning-gRPC/client/send_logs/logs/*')
				
					dis_time = time.time() - dis_stime
					try:
						lock.acquire(True)
						cur_learning.execute('''INSERT INTO DistributionTime VALUES (?, ?, ?);''', (ALL_CURRENT_ROUND-1, client_id, dis_time,))
						conn_learning.commit()
					finally:
						lock.release()
		elif req.version_req.type == 'P':
			client_id = change_clientname2index(req.version_req.config['client_name'].scstring)
			now_state = version_check(req.version_req.config['model_version'].scint32, req.version_req.config['current_round'].scint32)
			
			print(f"### [Version] client_id: {client_id} and delete_client_list: {delete_client_list} ###")
			if int(client_id) in delete_client_list:
				NUM_CLIENTS_CONTACTED_PER_ROUND -= 1
				cur_index.execute('''DELETE FROM ClientID WHERE id=?''', (client_id,))
				conn_index.commit()
				print(f"@@@ Delete client - {client_id} @@@")
				delete_rounds = [VersionRep(state=State.DELETE, config=dict())]
				for dr in delete_rounds:
					yield transportResponse(version_rep=dr)
			else:
				if now_state[0] == State.NOT_WAIT:
					state_dis_stime = time.time()
					send_params = list()
					with open('./server_weights/weights.pickle', 'rb') as fr:
						send_params = pickle.load(fr)
					for ns in [now_state]:
						yield transportResponse(version_rep=VersionRep(state=ns[0], buffer_chunk=pickle.dumps(send_params), config=ns[1]))
				elif now_state[0] == "FIN":
					for ns in [now_state]:
						yield transportResponse(version_rep=VersionRep(state=ns[0], config=ns[1]))
				else: ## "WAIT"
					for ns in [now_state]:
						yield transportResponse(version_rep=VersionRep(state=ns[0], config=ns[1]))

				if now_state[0] == State.NOT_WAIT:
					state_dis_time = time.time() - state_dis_stime
					try:
						lock.acquire(True)
						cur_learning.execute('''INSERT INTO DistributionTime VALUES (?, ?, ?);''', (ALL_CURRENT_ROUND-1, client_id, state_dis_time,))
						conn_learning.commit()
					finally:
						lock.release()

class TransportService(TransportServiceServicer):
	def transport(self, request, context):
		trans_res = manage_request(request)
		return trans_res

def serve():
	options = [('grpc.max_receive_message_length', 512*1024*1024), ('grcp.max_send_message_length', 512*1024*1024)]
	server = grpc.server(futures.ThreadPoolExecutor(max_workers=20), options=options)
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

	#eval_model = model_evaluate_for_mnist.evaluate_LocalModel(16, 48, np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))
	eval_model = model_evaluate.evaluate_LocalModel(16, 96, np.array(['0', '1', '2', '3', '4']))
	made_model = eval_model.buildGlobalModel(3, 0.001)
	serve()
