FROM python:3.8

MAINTAINER enter031@cs-cnu.org

RUN pip3 install numpy
RUN pip3 install pillow
RUN pip3 install tensorflow==2.4.0

RUN mkdir -p /app

WORKDIR /app

RUN mkdir -p server_weights
RUN mkdir -p common

COPY ./fl_server.py ./
COPY ./model_evaluate.py ./
COPY ./common/transport_pb2_grpc.py ./common
COPY ./common/transport_pb2.py ./common

EXPOSE 8890

CMD ["python3", "fl_server.py"]
