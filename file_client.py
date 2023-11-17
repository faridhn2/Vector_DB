import grpc
from file_transfer_pb2 import FileChunk
from file_transfer_pb2_grpc import FileTransferStub
import sys
def run_client(filename):
    with grpc.insecure_channel('localhost:50052') as channel:
        stub = FileTransferStub(channel)

        # Send chunks of data as messages
        for chunk_data in get_data(filename):
            stub.UploadFile(iter([FileChunk(data=chunk_data, filename=filename)]))

        print("File uploaded successfully")

def get_data(filename):
    # Implement your logic to generate data chunks (e.g., read from a file)
      # Set your desired chunk size
    with open(filename, 'rb') as content_file:
        content = content_file.read()
    yield content

if __name__ == '__main__':
    run_client(sys.argv[1])
