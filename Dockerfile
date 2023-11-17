FROM python:3.8
ADD requirements.txt /
RUN pip install -r /requirements.txt
ADD fa_nlp.py /
ADD file_server.py /
ADD file_client.py /
ADD file_transfer.proto /
ADD file_transfer_pb2.py /
ADD file_transfer_pb2_grpc.py /
ADD run.py /
ADD unary.proto /
ADD unary_client.py /
ADD unary_pb2.py /
ADD unary_pb2_grpc.py /
ADD unary_server.py /
ADD pdfs /
ENV PYTHONUNBUFFERED=1
CMD [ "python", "./run.py" ]
