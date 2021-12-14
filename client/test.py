from concurrent import futures

import os
import time
import json
import random
import pickle
import sqlite3
import threading
import requests
import numpy as np

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

delete_client_list = list()
time_client_list = list()
s_list = list()
tsd = 0
tsdk = 0
tkul = 0
tkud = 0
seta = 0
del_par_cli = 0
time_pass = False
cs_time = 0
taa = 0# sqlite3 database

conn_index = sqlite3.connect("./dashboard_db/index.db", check_same_thread=False)
cur_index = conn_index.cursor()
conn_learning = sqlite3.connect("./dashboard_db/learning.db", check_same_thread=False)
cur_learning = conn_learning.cursor()

cs_time = time.time()
del_par_cli += 1
clid = 1

cur_learning.execute('''SELECT uploadendtime-uploadstarttime AS sdis FROM LearningTime WHERE round=? and clientid=?''', (curr_oneround-1, clid,))
tkul = cur_learning.fetchone()[0]
cur_learning.execute('''SELECT trainingtime AS traint FROM LearningTrain WHERE round=? and clientid=?''', (curr_oneround-1, clid,))
tkud = cur_learning.fetchone()[0]
seta_t = seta + tkul + max(0, tkud - set)

for s in s_list:
	cur_learning.execute('''SELECT distributiontime AS sdis FROM DistributionTime WHERE round=? and clientid=?''', (curr_oneround-1, s,))
	tsd += cur_learning.fetchone()[0]
cur_learning.execute('''SELECT distributiontime AS sdis FROM DistributionTime WHERE round=? and clientid=?''', (curr_oneround-1, clid,))
tsdk = tsd + cur_learning.fetchone()[0]
cur_learning.execute('''SELECT aggregationtime AS sdis FROM AggregationTime WHERE round=? and clientid=?''', (curr_oneround-1, clid,))
taa = cur_learning.fetchone()[0]
t = (time.time() - cs_time) + tsd + seta_t + taa
cur_learning.execute('''SELECT uploadendtime-uploadstarttime AS sdis FROM LearningTime WHERE round=? and clientid=?''', (curr_oneround-2, clid,))
t1_upload = cur_learning.fetchone()[0]
cur_learning.execute('''SELECT trainingtime AS traint FROM LearningTrain WHERE round=? and clientid=?''', (curr_oneround-2, clid,))
t1_train = cur_learning.fetchone()[0]
cur_learning.execute('''SELECT distributiontime AS sdis FROM DistributionTime WHERE round=? and clientid=?''', (curr_oneround-2, clid,))
t1_dis = cur_learning.fetchone()[0]
cur_learning.execute('''SELECT aggregationtime AS sdis FROM AggregationTime WHERE round=? and clientid=?''', (curr_oneround-2, clid,))
t1_agg = cur_learning.fetchone()[0]

tround = t1_upload + t1_train + t1_dis + t1_agg
if t >= tround:
	seta = seta_t
	s_list.append(clid)	
