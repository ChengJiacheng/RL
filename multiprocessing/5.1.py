# -*- coding: utf-8 -*-

import multiprocessing as mp
import numpy as np
def f(x): #A
    return np.square(x)

if __name__ == '__main__':
    x = np.arange(64) #B
    print(x)
    print(mp.cpu_count())
    pool = mp.Pool(8) #C
    squared = pool.map(f, [x[8*i:8*i+8] for i in range(8)])
    pool.close()
    
    squared = np.asarray(squared)
    print(squared)