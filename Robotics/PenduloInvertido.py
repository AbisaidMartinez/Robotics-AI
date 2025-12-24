
import numpy as np 
import matplotlib.pyplot as plt
import sympy as sy
import matplotlib as mtlib
from matplotlib.patches import Rectangle
from matplotlib import animation
from numpy import linalg
import matplotlib.patheffects as pe

ci=[2,0,1,0]

l = sy.symbols('l')
t=sy.Symbol('t') #Variable simbolica tiempo
th = sy.Function('th')(t)   #Variable theta que depende del tiempo
x=sy.Function('x')(t)

x1= x
y1= 0
x2= x+(l*sy.sin(th))
y2=-l*sy.cos(th)

x1n=sy.lambdify(x,x1)
x2n=sy.lambdify((l,x,th),x2)
y2n=sy.lambdify((l,th),y2)

del l,t,th,x

mp=3
mc=5
l=1
g=9.81
fo=1
mp=2
mc=5
l=3

def f(t,x):#define la ec. dif. a resolver
    #[x[0],x[1],x[2],x[3]]
    #[x,v_x,y,v_y]

    x1=x[0]#x
    x2=x[1]#v_x
    x3=x[2]#y
    x4=x[3]#v_y
    
    dq=np.array([[x2],[x4]])  
    s=np.sin(x3)
    c=np.cos(x3)
    H=np.array([[(mp+mc),(mp*l*c)],[(mp*l*c),(mp*(l**2))]])
    H_inv=linalg.inv(H)
    C=np.array([[0,(-mp*x4*s)],[0,0]])
    Cq=np.matmul(C,dq)
    G=np.array([[0],[mp*g*l*s]])
    Bu=np.array([[fo],[0]])
    A=Bu-Cq-G
    D=np.matmul(H_inv,A)
    x_1p=x2
    x_2p=D[0]
    x_3p=x4
    x_4p=D[1]
    xp=np.hstack((x_1p,x_2p,x_3p,x_4p))
    return xp

def runge(a,b,N,ci):
    lista=[a]
    h=(b-a)/N
    p=h/2
    u=(1/6)*h
    M=ci
    xi=a
    c=0

    while c<N:
        k1=f(xi,ci)
        k2=f(xi+p,ci+(p*k1))
        k3=f(xi+p,ci+(p*k2))
        k4=f(xi+h,ci+(h*k3))
        ci+=((u*(k1+(2*k2)+(2*k3)+k4)))
        xi+=h
        c+=1
        lista.append(xi)
    
        M=np.vstack((M,ci))
    return lista,M
    
#         a   b     N
c,M=runge(0, 10, 1000, ci)

x1=M[:,0]#x
y1=M[:,1]#vx
x2=M[:,2]#th
y2=M[:,3]#vth

def xy(x,theta): #Evalua x,y
    return x1n(x),x2n(l,x,theta),y2n(l,theta)

t=np.linspace(0, 10,1000)

p_x1,p_x2,p_y2=xy(x1,x2)

x1_0=p_x1[0]
x2_0=p_x2[0]
y2_0=p_y2[0]


fig = plt.figure()
ax = fig.add_subplot(aspect='equal')
line,= ax.plot([x1_0, y2_0], [x2_0, y2_0], lw=1.5, c='white',path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
rect1=ax.add_patch(Rectangle(xy=(x1_0,0),width=3,height=1.5,color='darkred'))
rect2=ax.add_patch(Rectangle(xy=(x1_0,0),width=1.5,height=0.5,color='deepskyblue'))
circle1 = ax.add_patch(plt.Circle((x1_0,y2_0), 0.5, color='darkmagenta', zorder=2.5))
circle2 = ax.add_patch(plt.Circle((x1_0,y2_0), 0.2, color='k'))
circle3 = ax.add_patch(plt.Circle((x1_0,y2_0), 0.2, color='k'))
ax.set_xlim([0, 20])
ax.set_ylim([-5,5])

def animate(i):
    rect1.set_xy((p_x1[i],0))
    rect2.set_xy((p_x1[i]+1.5,1))
    line.set_data([p_x1[i]+1.5, p_x2[i]+1.5], [1.5,p_y2[i]])#Actualiza las coord. cada frame para el final de la linea
    circle1.set_center((p_x2[i]+1.5, p_y2[i]))#Actualiza las coord. cada frame para el centro del circulo
    circle2.set_center((p_x1[i]+0.5, 0))#Actualiza las coord. cada frame para el centro del circulo
    circle3.set_center((p_x1[i]+2.5, 0))#Actualiza las coord. cada frame para el centro del circulo



it = t[1]-t[0]
intervalo = it * 1000
ani = animation.FuncAnimation(fig, animate, frames=len(x1), repeat=True,interval=intervalo)#Los frames dependen de la cantidad de info. que se haya generado
plt.show() 



