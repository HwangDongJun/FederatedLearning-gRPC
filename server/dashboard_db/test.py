import sqlite3

conn_learning = sqlite3.connect("learning.db", check_same_thread=False)
cur_learning = conn_learning.cursor()

cur_learning.execute('''SELECT COUNT(distinct clientid) AS cc FROM LearningTrain''')
cc = cur_learning.fetchall()

print(cc)
