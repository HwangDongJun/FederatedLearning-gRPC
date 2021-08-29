FROM python:3.8

MAINTAINER enter031@cs-cnu.org

RUN pip3 install numpy
RUN pip3 install tensorflow==2.4.0

RUN mkdir -p /app

WORKDIR /app

RUN mkdir -p saved_weight
RUN mkdir -p send_logs
RUN mkdir -p common

COPY ./client_fit_model.py ./
COPY ./fl_client.py ./
COPY ./common/transport_pb2_grpc.py ./common
COPY ./common/transport_pb2.py ./common

CMD ["python3", "fl_client.py"]
