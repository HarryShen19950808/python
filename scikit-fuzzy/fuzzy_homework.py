# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:40:42 2018

@author: 沈鴻儒
"""

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# 微分
def diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

# 宇集合區間
x_start = -3
x_end = 3.1
x = np.arange(x_start, x_end, 0.3)

# 函數微分取max，再與epsilon相除
dg = diff(np.cos, x)
epsilon = 0.3
h1 = epsilon / np.around(max(dg))

# fuzzy set個數
cnt = int((x_end - x_start) / h1) + 1
A = {}
list_A = []
e = []

for i in range(1, cnt + 1):
    if i == 1:
        A["{%d}"%i] = fuzz.trimf(x, [-3, -3, -2.7])
        e.append(-3)
    elif i > 1 and i < 21:
        A["{%d}"%i] = fuzz.trimf(x, [-3 + 0.3*(i - 2), -3 + 0.3*(i - 1), -3 + 0.3*i])
        e.append(-3 + 0.3*(i - 1))
    elif i == 21:
        A["{%d}"%i] = fuzz.trimf(x, [2.7, 3, 3])
        e.append(3)
    plt.plot(x, A["{%d}"%i])
    
sum_A = (i - 1) * ((0.6 * 1)/2)

