# Runge Kutta 4to orden
import matplotlib.pyplot as plt
import matplotlib as mtlib
from matplotlib.patches import Rectangle
from matplotlib import animation
from numpy import linalg
import matplotlib.patheffects as pe
import numpy as np

k = 6
mc1 = 1
mc2 = 1
m = 10
l = 1
g = 9.81
ff = 1
w = np.sqrt(k/m)
# la funcion es dy/dx=2xy, pero se puede cambiar


def f(t, x):  # define la ec. dif. a resolver
    # [x[0],x[1],x[2],x[3]]
    # [x,v_x,theta,v_theta]

    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x_1p = x2
    x_2p = -m*w*x1/k+ff
    # x_2p=(1/(mc+mp*pow(np.sin(x3),2)))*(ff+mp*np.sin(x3)*(l*pow(x4,2)+g*np.cos(x3)));
    x_3p = 0
    x_4p = 0

    # x_4p=(1/(l*(mc+mp*pow(np.sin(x3),2))))*(-ff*np.cos(x3)-mp*l*pow(x4,2)*np.cos(x3)*np.sin(x3)-(mc+mp)*g*np.sin(x3));
    xp = np.hstack((x_1p, x_2p, x_3p, x_4p))
    return xp

# intervalo [a,b] a=t_0, N(iteraciones) y condicion (y(0)=?)


def RungeKutta(a, b, N, condicion):
    h = (b-a)/N
    lista1 = [a]
    x = a
    A = condicion
    L = condicion
    for i in range(N):
        k1 = f(x, A)
        k2 = f(x+h/2, A+(h/2)*k1)
        k3 = f(x+h/2, A+(h/2)*k2)
        k4 = f(x+h, A+h*k3)
        A += (h/6)*(k1+2*k2+2*k3+k4)
        # print("1");

        x += h
        # se capturan los datos en listas para graficar
        lista1.append(x)

        L = np.vstack((L, A))

    # print(lista1);
    # print(lista2);
        # print(x);
    return(lista1, L)


# primero manda a llamar a tu funcion, despues puedes graficar OJO
ci = [0.1, 0, 1, 0]
N = 1000
x, L = RungeKutta(0, 10, N, ci)
# print(x);

# Renombras para no tener confusiones con el manejo de listas
# Representas las ecuaciones de movimiento
x1 = L[:, 0]
y1 = L[:, 1]  # V_X
x2 = L[:, 2]
y2 = L[:, 3]

x2p = x1

# condiciones iniciales
x10 = x1[0]
y10 = y1[0]
x20 = x2p[0]


plt.close('all')
plt.style.use('classic')
fig = plt.figure()
ax = fig.add_subplot(aspect='equal')
# line, = ax.plot([x10, y20], [x20, y20], lw=1.5, c='white', path_effects=[
#     1.5+x10           pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
line, = ax.plot([min(x1), min(x1)], [10, 0], lw=1.5, c='black', path_effects=[
                pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
line2, = ax.plot([min(x1), 0.75], [0.75, 0.75], lw=3, c='white',
                 path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
rect1 = ax.add_patch(
    Rectangle(xy=(1.5+x10, 0), width=3, height=1.5, color='red'))

#circle1 = ax.add_patch(plt.Circle((x10, y20), 0.5, color='red', zorder=2.5))
#circle2 = ax.add_patch(plt.Circle((x10, y20), 0.2, color='k'))
#circle3 = ax.add_patch(plt.Circle((x10, y20), 0.2, color='k'))
ax.set_xlim([-1, 5])
ax.set_ylim([-1, 5])


def animate(i):
    line2.set_data([[min(x1), 0.75+x2p[i]], [0.75, 0.75]])
    rect1.set_xy((0.75+x2p[i], 0))
#    rect2.set_xy((x1[i]+1.5, 1))
#    rect3.set_xy((x3p[i]+4, 0))
#    rect4.set_xy((x3p[i]+5.5, 1))
    # Actualiza las coord. cada frame para el final de la linea
#    line.set_data([[x1[i]+1.5, x2p[i]+1.5], [1.5, y2p[i]]])
#    line1.set_data([[x3p[i]+2, x3p[i]+4.1], [0.75, 0.75]])


i = np.arange(0, 10, 1000)
# Los frames dependen de la cantidad de info. que se haya generado
ani = animation.FuncAnimation(
    fig, animate, frames=len(x1), repeat=True, interval=i)
ani.save('OsciladorArmonico.gif')
plt.show()
