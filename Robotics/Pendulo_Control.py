# Runge Kutta 4to orden
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patheffects as pe

#import sympy as sy
#import matplotlib as mtlib
#from numpy import linalg
#from matplotlib.animation import PillowWriter


delta=-np.pi/2;

l=.1978;
g=9.81;
m=.03626;
disipacion=0.47;

A=g/l;
B=disipacion/(m*pow(l,2));
C=1/(m*pow(l,2));

K1=10;
K2=1;
K3=(1/10)*K1;


def f(t, x):  # define la ec. dif. a resolver
    # [x[0],x[1],x[2],x[3]]
    # [x,v_x,y,v_y]
    
    # La ecuacion del pendulo es theta_pp=-a*sin(theta)-b*theta+c*T

    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    v=-K1*x1-K2*x2-K3*x3;
    u=(1/C)*(A*(np.sin(x1+delta)-np.sin(delta))+B*x2+v);
    
    x_1p = x2
    x_2p = -A*(np.sin(x1+delta)-np.sin(delta))-B*x2+C*u;
    x_3p = x1  
    x_4p = 0  
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

        x += h
        

        # se capturan los datos en listas para graficar

        lista1.append(x)

        L = np.vstack((L, A))

    return(lista1, L)


# primero manda a llamar a tu funcion, despues puedes graficar OJO
ci = [0, 5*np.pi/2, 0, 0]
x, L = RungeKutta(0, 50, 5000, ci)
# print(x);

x1 = L[:, 0]-delta
y1 = L[:, 1]  # V_X
x2 = L[:, 2]
y2 = L[:, 3]


x1p = l*np.sin(x1)
y1p = -l*np.cos(x1)

# condiciones iniciales
x10 = x1[0]
y10 = y1[0]

# Animación del péndulo
plt.close('all')
plt.style.use('classic')
fig = plt.figure()
ax = fig.add_subplot(aspect='equal')
line, = ax.plot([0, x10], [0, y10], lw=1.5, c='white', path_effects=[
                pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])

circle1 = ax.add_patch(plt.Circle((x10, y10), 0.02, color='red', zorder=2.5))
ax.set_xlim([-0.25, 0.25])
ax.set_ylim([-.25, .25])


def animate(i):
    
    # Actualiza las coord. cada frame para el final de la linea
    line.set_data([0, x1p[i]], [0, y1p[i]])

    # Actualiza las coord. cada frame para el centro del circulo
    circle1.set_center((x1p[i], y1p[i]))
    return line, circle1

i = np.arange(0, 10, 500)
# Los frames dependen de la cantidad de info. que se haya generado
ani = animation.FuncAnimation(
    fig, animate, frames=len(x1), repeat=True, interval=i)
#ani.save('Control.gif', writer=PillowWriter(fps=20))
plt.show()
