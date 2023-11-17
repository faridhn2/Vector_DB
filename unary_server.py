import grpc
from concurrent import futures
import time
import unary_pb2_grpc as pb2_grpc
import unary_pb2 as pb2
from fa_nlp import MyVectorDB
vdb = None
class UnaryService(pb2_grpc.UnaryServicer):

    def __init__(self, *args, **kwargs):
        global vdb
        vdb = MyVectorDB()
    def GetServerResponse(self, request, context):
        
        # get the string from the incoming request
        message = request.message

        if str(message) == 'restart':
          result = 'restarted'
          vdb.restart()
          result = {'message': result, 'received': True}
        elif str(message) == 'list':
          result = vdb.get_doc_list()
          
          result = {'message': result, 'received': True}
        elif 'search' in str(message):
          search_key = str(message).replace('search','')
          result = vdb.search(search_key)
          
          result = {'message': result , 'received': True}
        elif 'summary' in str(message):
         
            summary_idx = int(str(message).split()[1])
            result = vdb.get_summary(summary_idx)
            # result = summary_idx
            result = {'message': result, 'received': True}
          
        else:
          result = 'Wrong'
          result = {'message': result, 'received': True}
        return pb2.MessageResponse(**result)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_UnaryServicer_to_server(UnaryService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()