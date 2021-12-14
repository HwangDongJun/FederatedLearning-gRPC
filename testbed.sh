#!bin/bash

SET=$(seq 0 10)
for i in $SET
do
	sudo docker run -d --name fl_client_$i --network host fl_client:1.0
done
