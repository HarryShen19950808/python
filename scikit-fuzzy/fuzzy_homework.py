# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:40:42 2018

@author: 沈鴻儒
"""

import numpy as np
import matplotlib.pyplot as plt

# 建立membership function 的 fuzzy set
def trimf(x, abc):
    """
    Triangular membership function generator.

    Parameters
    ----------
    x : 1d array
        Independent variable.
    abc : 1d array, length 3
        Three-element vector controlling shape of triangular function.
        Requires a <= b <= c.

    Returns
    -------
    y : 1d array
        Triangular membership function.
    """
    assert len(abc) == 3, 'abc parameter must have exactly three elements.'
    a, b, c = np.r_[abc]     # Zero-indexing in Python
    assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

    y = np.zeros(len(x))

    # Left side
    if a != b:
        idx = np.nonzero(np.logical_and(a < x, x < b))[0]
        y[idx] = (x[idx] - a) / float(b - a)

    # Right side
    if b != c:
        idx = np.nonzero(np.logical_and(b < x, x < c))[0]
        y[idx] = (c - x[idx]) / float(c - b)

    idx = np.nonzero(x == b)
    y[idx] = 1
    return y 

# 建立fuzzy set的個數
def fuzzy_set(x, x_start, x_end, cnt, epsilon):
    global e
    for i in range(1, cnt+1):
        if i == 1:
            A["{%d}"%i] = trimf(x, [x_start, x_start, -3 + epsilon*(i)])
            e.append(-3)
        elif i > 1 and i < cnt:
            A["{%d}"%i] = trimf(x, [-3 + epsilon*(i - 2), -3 + epsilon*(i - 1), -3 + epsilon*i])
            e.append(-3 + epsilon*(i - 1))
        elif i == cnt:
            A["{%d}"%i] = trimf(x, [-3 + epsilon*(i - 2), x_end, x_end])
            e.append(3)
            e = np.array(e).reshape(cnt, -1)
    return A, e
# 微分
def diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def g(x):
    return np.cos(x)
# 定義宇集合區間與epsilon
epsilon = 0.3 # 
x_start = -3
x_end = 3.000001
x = np.arange(x_start, x_end, epsilon)
x_for_g = np.arange(x_start, x_end, 0.01)

# 函數微分取max，再與epsilon相除
dg = diff(g, x)
h1 = epsilon / np.around(max(dg))

# 利用h計算fuzzy set個數
cnt = int((x_end - x_start) / h1) + 1
# fuzzy set
A = {}

# 收集fuzzy set中值
e = []

fuzzy_set(x, x_start, x_end, cnt, epsilon)
for i in range(1, cnt+1):
    plt.figure(1)
    plt.plot(x, A["{%d}"%i])
    f = (g(e) * A["{%d}"%i]) / A["{%d}"%i]

plt.figure(2)
plt.plot(x_for_g, g(x_for_g), label = "g(x)")
plt.plot(x, f, label = "f(x)")
plt.legend()
plt.show()
