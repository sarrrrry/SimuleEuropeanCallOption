import matplotlib.pyplot as plt
import numpy as np

h = 0.01


def fn(t):
    """solution of differential equation"""
    return t ** 2


def ode(t, y):
    """ordinary differential equation"""
    dy = (2 * y) / t
    return dy


def eular_method(y, dy):
    """next y calculated by Euler-method"""
    return (h * dy) + y


x_true = np.linspace(-10, 10, 1000)

t1 = 1
y1 = 1

t = [t1, ]
y = [y1, ]
while t[-1] < x_true.max():
    dy = ode(t[-1], y[-1])
    y_dh = eular_method(y[-1], dy)

    y.append(y_dh)
    t.append(t[-1] + h)
while t[-1] > x_true.min():
    dy = ode(t[-1], y[-1])
    y_dh = eular_method(y[-1], -dy)

    y.append(y_dh)
    t.append(t[-1] - h)

plt.plot(x_true, fn(x_true), label="true fn")
plt.scatter(t, y, label="sim fn", color='r')
plt.legend()
plt.show()
