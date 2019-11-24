# -*- coding: utf-8 -*-
import numpy as np

def square(i, x, queue):
    print("In process {}".format(i,))
#    sys.stdout.flush()
    queue.put(np.square(x))