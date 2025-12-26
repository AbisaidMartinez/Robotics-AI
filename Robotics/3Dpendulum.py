import numpy as np 
import sympy as sy
#from scipy.integrate import odeint,RK45
import matplotlib.pyplot as plt
from matplotlib import animation
import vpython as vp
from vpython import *

ci=[1 , 0 ,1 ,0 ]

m,g,l = sy.symbols(('m', 'g','l'))
t=sy.Symbol('t') #Variable simbolica tiempo
th = sy.Function('th')(t)   #Variable theta que depende del tiempo
vth=th.diff(t)
ath=vth.diff(t)
ap=sy.Function('ap')(t)
vap=ap.diff(t)
aap=vap.diff(t)


x= l*sy.sin(th)*sy.cos(ap)
y= l*sy.sin(th)*sy.sin(ap)
z=-l*sy.cos(th)

T=(1/2)*m*((x.diff(t)**2)+(y.diff(t)**2)+(z.diff(t)**2)) #E. Cinetica
V=m*g*z #E. Potencial
L= T-V #Lagrangiano

dL_theta=L.diff(th) #Parcial de L respecto a posicion
dL_vtheta_t=sy.diff(L.diff(vth),t) #Parcial de L respecto a velocidad

dL_ap=L.diff(ap) #Parcial de L respecto a posicion
dL_vap_t=sy.diff(L.diff(vap),t) #Parcial de L respecto a velocidad

eq1=dL_vtheta_t-dL_theta
eq2=dL_vap_t-dL_ap

Eu_La1=sy.solve(eq1,ath)[0] #Despeja la ecuacion algebraicamente
Eu_La2=sy.solve(eq2,aap)[0] #Despeja la ecuacion algebraicamente

print(Eu_La2)

                  #variables     #ecuacion
dth_n=sy.lambdify(vth,                vth)#convierte valores simbolicos a numericos
dap_n=sy.lambdify(vap,                vap)#convierte valores simbolicos a numericos
eulerl1=sy.lambdify((l,g,vap,th),      Eu_La1)
eulerl2=sy.lambdify((l,g,vap,th,vth),      Eu_La2)
xn=sy.lambdify(    (l,th,ap),                 x)
yn=sy.lambdify(    (l,th,ap),                 y)
zn=sy.lambdify(    (l,th),                 z)

mp=3
mc=5
l=1
g=9.81
fo=1

def f(t,x):#define la ec. dif. a resolver
    #[x[0],x[1],x[2],x[3]]
    #[x,v_x,y,v_y]

    x1=x[0]#x
    x2=x[1]#v_x
    x3=x[2]#y
    x4=x[3]#v_y

    x_1p=x2
    x_2p=eulerl1(l,g,x4,x1)
    x_3p=x4
    x_4p=eulerl2(l,g,x4,x1,x2)
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

x1=M[:,0]#th
y1=M[:,1]#vth
z1=M[:,2]#ap
"""
plt.close('all')
fig, ax = plt.subplots()
#ax.plot(x, L[:,0],'--') 
#ax.scatter(L[:,0],L[:,1]) 
plt.title("Orbita (fi vs Vfi)")
ax.set_xlabel('x(t)')
ax.set_xlabel('v_x(t)')
plt.plot(M[:,0],M[:,1])   

fig2, ax2 = plt.subplots()
ax2.plot(M[:,2],M[:,3]) 
plt.title("Orbita (th vs Vth)")
ax.set_xlabel('x(t)')
ax.set_xlabel('v_x(t)')

fig3, ax3 = plt.subplots()
ax3.plot(c,M[:,0]) 
#ax2.plot(x,L[:,2],label=r'$x_2$') 
#ax2.plot(x,L[:,3],label=r'$x_3$') 
plt.title("fi vs t")
plt.xlabel("t (s)")
plt.ylabel("y(t)")

fig3, ax4 = plt.subplots()
ax4.plot(c,M[:,2]) 
#ax2.plot(x,L[:,2],label=r'$x_2$') 
#ax2.plot(x,L[:,3],label=r'$x_3$') 
plt.title("th vs t")
plt.xlabel("t (s)")
plt.ylabel("y(t)")
plt.show()
"""
def xy(alpha,theta): #Evalua x,y
    return xn(l,theta,alpha),yn(l,theta,alpha),zn(l,theta)

t=np.linspace(0, 10,1000)

p_x1,p_x2,p_y2=xy(z1,x1)

x1_0=p_x1[0]
x2_0=p_x2[0]
y2_0=p_y2[0]

ball1 = vp.sphere(color = color.green, radius = 0.3, make_trail=True, retain=20)
rod1 = cylinder(pos=vector(0,0,0),axis=vector(0,0,0), radius=0.05)
base  = box(pos=vector(0,-6,0),axis=vector(1,0,0),size=vector(10,0.5,10) )


i = 0
while True:
    rate(30)
    i = i + 1
    i = i % len(x1)
    ball1.pos = vector(x1[i], -z1[i], y1[i])
    rod1.axis = vector(x1[i], -z1[i] , y1[i])














