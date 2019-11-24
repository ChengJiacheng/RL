# -*- coding: utf-8 -*-
import multiprocessing as mp
import numpy as np
import sys

def square(i, x, queue):
    print("In process {}".format(i,))
#    sys.stdout.flush()
    queue.put(np.square(x))
    
if __name__ == '__main__':
    processes = [] #A
    queue = mp.Queue() #B
    x = np.arange(64) #C
    for i in range(8): #D start 8 processes
        start_index = 8*i
        proc = mp.Process(target=square,args=(i,x[start_index:start_index+8], queue)) 
        proc.start()
        processes.append(proc)
        
    for proc in processes: #E 
        proc.join()
        
    for proc in processes: #F
        proc.terminate()
    results = []
    while not queue.empty(): #G
        results.append(queue.get())
    
    results = np.asarray(results)
    print(results)
    
    k=input("press any key to exit") 
