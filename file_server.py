import grpc
from concurrent import futures
from file_transfer_pb2 import FileResponse, FileChunk
from file_transfer_pb2_grpc import FileTransferServicer, add_FileTransferServicer_to_server

class FileTransferService(FileTransferServicer):
    def UploadFile(self, request_iterator, context):
        received_data = b""
        for request in request_iterator:
            received_data += request.data  # Accumulate the chunks
            filename = request.filename
        with open(f'pdfs/{filename}','wb') as f:
            f.write(received_data)    
        # Process or save the received data
        print("Received ")

        return FileResponse(message="File uploaded successfully")

def run_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_FileTransferServicer_to_server(FileTransferService(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    run_server()
