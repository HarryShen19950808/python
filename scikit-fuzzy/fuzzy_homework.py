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
def fuzzy_set(x, x_start, x_end, cnt, epsilon, A, e, h1):
    for i in range(cnt):
        if i == 0:
            A[i, :, :] = trimf(x, [x_start, x_start, x_start + h1]).reshape(-1, 1)
            e.append(x_start)
        elif i > 0 and i < cnt:
            A[i, :, :] = trimf(x, [x_start + (h1 * (i - 1)), x_start + (h1 * i), x_start + (h1 * (i + 1))]).reshape(-1, 1)
            e.append(x_start + (h1 * i))
        elif i == cnt:
            A[i, :, :] = trimf(x, [x_start + (h1 * (i - 2)), x_end, x_end]).reshape(-1, 1)
            e.append(x_end)

# 微分
def diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# 目標函數
def g(x):
    return np.cos(x)


# 定義宇集合區間與epsilon
epsilon = 0.3 # 
x_start = -3
x_end = 3
x = np.linspace(x_start, x_end, 1000)

# g(x)微分取sup，並計算出h1
dg = diff(g, x)
h1 = epsilon / np.around(max(dg))

# 計算fuzzy set個數
cnt = int(((x_end - x_start) / h1) + 1)

# 計算fuzzy set
A = np.empty((cnt, int(len(x)), 1))
e = []

fuzzy_set(x, x_start, x_end, cnt, epsilon, A, e, h1)

for k in range(cnt):
    plt.figure(1)
    plt.plot(x, A[k])
    for j in range(1, len(x)):
        f = (g(e) * A[k][j]) / (A[k][j])

plt.figure(2)
plt.plot(e, f, label = "f(x)")
plt.plot(e, g(e), label = "g(x)")
plt.legend()
plt.show()