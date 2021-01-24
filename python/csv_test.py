#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:28:47 2020

@author: philipp
"""

import numpy as np


hist1 = [np.array([1,1,1])for n in range(10)]



# np.savetxt("/Users/philipp/Documents/GitHub/stage_cmap/python/locations.csv",  
#            hist1, 
#            delimiter =", ",  
#            fmt ='% s')

hist1_read = np.genfromtxt('/Users/philipp/Documents/GitHub/stage_cmap/python/locations.csv'\
                           , delimiter=',')
    
print(hist1_read)

filepath = "/Users/philipp/Documents/GitHub/stage_cmap/python/"



