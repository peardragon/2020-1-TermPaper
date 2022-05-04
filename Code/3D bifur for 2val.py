import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def func(r, v, x):

    return x*(1-x)*(r-v*x)


def bifurcation_figure(vvalmin, vvalmax, rmin, rmax, ymin, ymax, iterations, last, n, n1, x0):
    r = np.linspace(rmin, rmax, n)
    # r 사이 간격 (찍히는 점의 개수)

    # Identity list with n

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    fig2 = plt.figure()
    ax1 = fig2.gca()
    v0 = np.linspace(vvalmin, vvalmax, n1)
    for i in range(n1):
        x = x0 * np.ones(n)
        v = v0[i] * np.ones(n)
        ini = v0[i] * np.ones(n)
        for i in range(iterations):
            x = func(r, v, x) # x1 = f(x0) 계산, 반복
            # We display the bifurcation diagram.
            if i >= (iterations - last):
                ax.plot(ini, r, x, ',k', ms=5, alpha=0.125)
                ax1.plot(ini, r, 0 , ',k', ms=5, alpha=0.125)
            # 충분히 진행한다음 후반 last번째 값, 수렴하면 하나로 찍히겠지만, 안할경우 chaos적 점이 찍힘
    ax.set_zlim(ymin, ymax)
    ax.set_xlim(vvalmin, vvalmax)
    ax.set_ylim(rmin, rmax)
    ax.set_xlabel('r')
    ax.set_ylabel('v')
    ax.set_zlabel('x')
    ax.set_title(f"$Bifurcation diagram$")

    plt.show()
    plt.show()
'''
def bifurcation_figure(xvalmin, xvalmax, rmin, rmax, ymin, ymax, iterations, last, n, n1):
    r = np.linspace(rmin, rmax, n)
    # r 사이 간격 (찍히는 점의 개수)

    # Identity list with n

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x0 = np.linspace(xvalmin, xvalmax, n1)
    for i in range(n1):
        x = x0[i] * np.ones(n)
        ini = x0[i] * np.ones(n)
        for i in range(iterations):
            x = func(r, x)# x1 = f(x0) 계산, 반복
            # We display the bifurcation diagram.
            if i >= (iterations - last):
                ax.plot(ini, r, x, ',k', ms=5, alpha=0.125)
            # 충분히 진행한다음 후반 last번째 값, 수렴하면 하나로 찍히겠지만, 안할경우 chaos적 점이 찍힘
    ax.set_zlim(ymin, ymax)
    ax.set_xlim(xvalmin, xvalmax)
    ax.set_ylim(rmin, rmax)
    ax.set_xlabel('x0')
    ax.set_ylabel('r')
    ax.set_zlabel('x')
    ax.set_title(f"$Bifurcation diagram$")
    plt.show()
'''


# bifurcation_figure(xvalmin, xvalmax, rmin, rmax, ymin, ymax, iterations, last, n,n1,x0):
bifurcation_figure(0, 8, 0, 8, 0, 1, 100, 10, 1000, 20, 0.2)
