import unittest
import os
import multiprocessing
import os
import time
from file_client import run_client as file_uploader


def file_s():
  os.system('python file_server.py')

class TestBasic(unittest.TestCase):
    def setUp(self):
        
        p2 = multiprocessing.Process(target=file_s)
        p2.start()
        time.sleep(30)
    def test_upload(self):
        files=  ['ml.pdf','PCA.pdf','Tree.pdf']
        for file in files:
          file_uploader(file)
          pdf_list = os.listdir('pdfs')
          outp = file in pdf_list
          self.assertEqual(outp, True)

    
if __name__ == '__main__':
    unittest.main()