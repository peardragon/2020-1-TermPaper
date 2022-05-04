import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def func(r, x):
    return x+r*np.sin(x)



def diff(r, x):
    t = sp.Symbol('t')
    y = func(r, t)

    fnc = y.diff(t)

    return fnc.subs(t, x)


def d_func(r, x):
    h = 1e-5
    return (func(r, x+h)-func(r, x-h))/(2*h)


def system_figure(r, x0, n, tval,ax=None):
    # Plot the function and the
    # y=x diagonal line.
    t = np.linspace(-tval, tval)
    # Defalut  50개의 배열
    ax.plot(t, func(r, t), 'k', lw=2)
    # lw 선굵기
    ax.plot([-tval, tval], [-tval, tval], 'k', lw=2)

    # Recursively apply y=f(x) and plot two lines:
    # (x, x) -> (x, y)
    # (x, y) -> (y, y)
    x = x0
    for i in range(n):
        y = func(r, x)
        # Plot the two lines.
        ax.plot([x, x], [x, y], 'k', lw=1)
        ax.plot([x, y], [y, y], 'k', lw=1)
        # Plot one web part. x line & y line
        # Plot the positions with increasing
        # opacity.
        ax.plot([x], [y], 'ok', ms=10,
                alpha=(i + 1) / n)
        # ms 마커크기 , alpha 투명도 / 0 이면 완전투명
        x = y
        # x = y 를 따라 x, y 재설정

    ax.set_xlim(-tval, tval)  # 축설정, y 0 ~ 1 까지
    ax.set_ylim(-tval, tval)
    ax.set_title(f"$r={r:.1f}, \, x_0={x0:.1f}$")


# *Default bifurcation diagram 2,8 to 4
def bifurcation_figure(x0, xmin, xmax, ymin, ymax, iterations, last, n):
    r = np.linspace(xmin, xmax, n)
    # r 사이 간격 (찍히는 점의 개수)
    x = x0 * np.ones(n)
    # Identity list with n
    nlyapunov = np.zeros(n)
    # Zero list with n

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=False)
    for i in range(iterations):
        x = func(r, x) # x1 = f(x0) 계산, 반복
        # We compute the partial sum of the
        # n times with Lyapunov exponent.
        nlyapunov += np.log(abs(d_func(r, x)))
        # lyapunov = 1/n * Sum of log {f(x_i)} (r-2rx) is derivative of func fnc.

        #Have to change this part to derivative some func.
        # and += mean sum for iterations

        # We display the bifurcation diagram.
        if i >= (iterations - last):
            ax1.plot(r, x, ',k', ms=5, alpha=0.125)
            # 충분히 진행한다음 후반 last번째 값, 수렴하면 하나로 찍히겠지만, 안할경우 chaos적 점이 찍힘

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlabel('r')
    ax1.set_ylabel('x')
    ax1.set_title(f"$Bifurcation diagram, \, x_0={x0:.1f}$")

    # We display the Lyapunov exponent.
    # Horizontal line.
    lyapunov = nlyapunov

    ax2.axhline(0, color='k', lw=.5, alpha=.5)
    # Negative Lyapunov exponent.
    # r[condition], lyapunov[conditon]
    ax2.plot(r[lyapunov < 0], lyapunov[lyapunov < 0] / iterations, '.k', alpha=.5, ms=.5)
    # Positive Lyapunov exponent.
    ax2.plot(r[lyapunov >= 0], lyapunov[lyapunov >= 0] / iterations, '.r', alpha=.5, ms=.5)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(-2, 1)
    ax2.set_xlabel('r')
    ax2.set_ylabel('x')
    ax2.set_title(f"$Lyapunov exponent, \, x_0={x0:.1f}$")

    plt.tight_layout()


def iteration_figure(x0, r0, ymin, ymax, last, start, final):

    # r 사이 간격 (찍히는 점의 개수)
    x = x0
    r = r0
    # 10^(-5)*Identity list with n
    val = []
    val.append(x0)
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 9), sharex=True)
    for i in range(final):
        #for i in 300 - 80 = 220 0 to 220
        val.append(func(r, val[i]))
        # val - length : final, list x0 x1 ... x300
    Range = final - start
    for j in range(Range):
        if j>= last:
            ycord = val[j + start - last: j + start]
            xcord = [j + start] * (last)
            ax1.plot(xcord, ycord, '.k', ms=3, alpha=1)

    ax1.set_xlim(start, final)
    ax1.set_ylim(ymin, ymax)
    ax1.set_title("Iteration diagram")

    plt.show()


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


'''
    for i in range(n-1):
        if lyapunov[i]*lyapunov[i+1] <= 0:
            a = xmin+(xmax-xmin)*(i/n)
            print(a)
'''
'''
x = np.linspace(0, 1)
fig, ax = plt.subplots(1, 1)
# fig : 전체 서브 플롯, ax : 낱낱개
ax.plot(x, func(2, x), 'k')
'''
'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
# system_figure(r, x0, n, ax=None)

system_figure(2.5, .1, 10, ax=ax1)
# ax=ax1에 fig 적용
system_figure(3.5, .1, 10, ax=ax2)
# ax=ax2에 fig 적용
'''

#bifurcation_figure(0.2, 2.8, 4, 0, 1, 1000, 100, 7)
#bifurcation_figure(0.2, 2.8, 4, 0, 1, 1000, 100, 10000)
#bifurcation_figure(0.2,3.840, 3.857, 0.44, 0.55, 10000, 1000, 10000)
#bifurcation_figure(0.2,3.8470, 3.8501, 0.46, 0.474, 10000, 1000, 10000)
#def system_figure(r, x0, n, tval,ax=None):
fig, (ax5,ax4) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
system_figure(2.8, 0.2, 80,6, ax=ax5)
system_figure(3.2, 0.2, 160,6, ax=ax4)
# system figure ( r, x0, iteration, tval,  ax)
# bifurcation_figure(x0, xmin, xmax, ymin, ymax, iterations, last, n)
#bifurcation_figure(0.2, 2.8, 4, 0, 1, 10000, 1000, 100000)
#ifurcation_figure(0.2,3.840, 3.857, 0.44, 0.55, 1000, 100, 10000)
bifurcation_figure(0.2, 0, 6, 0, 6, 1000, 100, 10000)

#iteration_figure(0.51, 3.82831, 0, 1, 3, 0, 500)

plt.show()