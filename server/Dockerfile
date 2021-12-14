FROM python:3.8

MAINTAINER enter031@cs-cnu.org

RUN apt update -y
RUN apt install nodejs -y
RUN apt install npm -y
RUN pip3 install numpy
RUN pip3 install pillow
RUN pip3 install tensorflow==2.4.0

RUN mkdir -p /app

WORKDIR /app

RUN mkdir -p server_weights
RUN mkdir -p common
RUN mkdir -p dashboard_db
RUN mkdir -p FLDashboard

COPY ./fl_server_dashboard.py ./
COPY ./model_evaluate_for_mnist.py ./
#COPY ./fl_server.py ./
#COPY ./model_evaluate_for_mnist.py ./
COPY ./common/transport_pb2_grpc.py ./common
COPY ./common/transport_pb2.py ./common
COPY ./for_docker_dashboard/index.db ./dashboard_db
COPY ./for_docker_dashboard/learning.db ./dashboard_db
COPY ./dashboard/. ./FLDashboard

EXPOSE 8891
EXPOSE 5003

CMD ["python3", "fl_server_dashboard.py"]
