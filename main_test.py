import unittest
import os
import multiprocessing
import os
import time

from unary_client import UnaryClient
def main_server():
  os.system('python unary_server.py')


class TestBasic(unittest.TestCase):
    def setUp(self):
        p = multiprocessing.Process(target=main_server)
        print('Wait two minutes')
    #   print('Running main server')
        p.start()
        time.sleep(90)
        

    
    # def test_restart(self):
    #     client = UnaryClient()
    #     result = client.get_url(message='restart')
    #     time.sleep(300)
    #     self.assertEqual(result.message, "restarted")
    #     self.assertEqual(result.received, True)
    
    def test_list(self):
        client = UnaryClient()
        result = client.get_url(message='list')
        self.assertEqual(result.message, "0 - DataML_Engineer_Assignment.pdf \n1 - Intro to K Means Clustering.pdf \n2 - Python - Intro to Linear Regression.pdf \n")
        self.assertEqual(result.received, True)

    
    def test_search1(self):
        client = UnaryClient()
        result = client.get_url(message="search 'What is a good clustering method ?'")
        self.assertEqual(result.message, "Python - Intro to Linear Regression.pdf\nIntro to K Means Clustering.pdf")
        self.assertEqual(result.received, True)

    def test_search2(self):
        client = UnaryClient()
        result = client.get_url(message="search 'I need a Regression model'")
        self.assertEqual(result.message, "Intro to K Means Clustering.pdf\nPython - Intro to Linear Regression.pdf")
        self.assertEqual(result.received, True)


if __name__ == '__main__':
    unittest.main()