import grpc
import unary_pb2_grpc as pb2_grpc
import unary_pb2 as pb2
import sys

class UnaryClient(object):
    """
    Client for gRPC functionality
    """

    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051

        # instantiate a channel
        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port))

        # bind the client and the server
        self.stub = pb2_grpc.UnaryStub(self.channel)

    def get_url(self, message):
        """
        Client function to call the rpc for GetServerResponse
        """
        message = pb2.Message(message=message)
        print(f'{message}')
        return self.stub.GetServerResponse(message)


if __name__ == '__main__':
    try:
        client = UnaryClient()
        result = client.get_url(message=' '.join(sys.argv[1:]))
        print(f'{result.message}')  
    except:
        print('Wait for server. Try 10 Seconds later')