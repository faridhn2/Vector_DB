import multiprocessing
import os
import time
def main_server():
  os.system('python unary_server.py')
p = multiprocessing.Process(target=main_server)
print('Wait two minutes')
print('Running main server')
p.start()
time.sleep(120)
def file_s():
  os.system('python file_server.py')
p2 = multiprocessing.Process(target=file_s)
p2.start()
print('Running file server')
while True:
    try:
        command = input('Type Your Cammand: ')
        if 'file' not in command:
            os.system(f'python unary_client.py {command}')
        else:
            command = command.replace('file ','')
            os.system(f'python unary_client.py {command}')
    except:
        pass