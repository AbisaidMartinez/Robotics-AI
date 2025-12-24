import sympy as sp
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Definir símbolos
a, b = sp.symbols('a b')
x = sp.Matrix([a, b])

# Medias y covarianzas como matrices SymPy para precisión simbólica
mu_1 = sp.Matrix([3, 6])
mu_2 = sp.Matrix([3, -2])
sigma_1 = sp.Matrix([[sp.Rational(1,2), 0], [0, 2]])
sigma_2 = sp.Matrix([[2, 0], [0, 2]])

# Inversas simbólicas
inv_sig1 = sigma_1.inv()
inv_sig2 = sigma_2.inv()

pclass1 = sp.Rational(1,2)
pclass2 = sp.Rational(1,2)

# Componentes para sg1
W1 = -sp.Rational(1,2) * inv_sig1
w1 = inv_sig1 * mu_1
w10 = -sp.Rational(1,2) * (mu_1.T * inv_sig1 * mu_1)[0] - sp.Rational(1,2) * sp.log(sigma_1.det()) + sp.log(pclass1)

# Componentes para sg2
W2 = -sp.Rational(1,2) * inv_sig2
w2 = inv_sig2 * mu_2
w20 = -sp.Rational(1,2) * (mu_2.T * inv_sig2 * mu_2)[0] - sp.Rational(1,2) * sp.log(sigma_2.det()) + sp.log(pclass2)

# Expresiones discriminantes escalares (extrayendo el escalar de las matrices 1x1)
quad1 = (x.T * W1 * x)[0]
lin1 = (w1.T * x)[0]
sg1_scalar = quad1 + lin1 + w10

quad2 = (x.T * W2 * x)[0]
lin2 = (w2.T * x)[0]
sg2_scalar = quad2 + lin2 + w20

sg = sg1_scalar - sg2_scalar

# Frontera de decisión: resolver sg = 0 para b en términos de a
dec_bound = sp.solve(sg, b)  # Lista con la solución

# Lambdify para evaluación numérica
sg1_func = sp.lambdify((a, b), sg1_scalar, 'numpy')
sg2_func = sp.lambdify((a, b), sg2_scalar, 'numpy')
dec_bound_func = sp.lambdify(a, dec_bound[0], 'numpy')  # Asumiendo una solución única

# Gráfica 1: Frontera de decisión
plt.figure(1)
a_vals = np.linspace(-2, 10, 1000)
b_vals = dec_bound_func(a_vals)
plt.plot(a_vals, b_vals)
plt.axvline(0, color='black')
plt.axhline(0, color='black')
plt.xlim([-2, 10])
plt.ylim([-2.5, 10])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Decision Boundary")
plt.grid(True)
plt.show()

# Generar datos aleatorios (aquí usamos NumPy para eficiencia)
n = 100
mu_1_np = np.array([3, 6])
mu_2_np = np.array([3, -2])
sigma_1_np = np.array([[0.5, 0], [0, 2]])
sigma_2_np = np.array([[2, 0], [0, 2]])
data_class1 = multivariate_normal.rvs(mean=mu_1_np, cov=sigma_1_np, size=n)
data_class2 = multivariate_normal.rvs(mean=mu_2_np, cov=sigma_2_np, size=n)

x1 = data_class1[:, 0]
y1 = data_class1[:, 1]
x2 = data_class2[:, 0]
y2 = data_class2[:, 1]

g1 = np.zeros(len(x1))
g2 = np.zeros(len(x2))
vect = []  # Lista que contiene las asignaciones

# Evaluación de datos (réplica fiel del bucle original)
for i in range(len(x1)):
    fg1 = sg1_func(x1[i], y1[i])
    fg2 = sg2_func(x2[i], y2[i])
    g1[i] = fg1
    g2[i] = fg2
    if g1[i] > g2[i]:
        vect.append("w1")
    else:
        vect.append("w2")

sum_w1 = vect.count("w1")
sum_w2 = vect.count("w2")
percent_w1 = (sum_w1 / len(vect)) * 100
percent_w2 = (sum_w2 / len(vect)) * 100

print(f"percent_w1: {percent_w1}")
print(f"percent_w2: {percent_w2}")

# Gráfica 2
plt.figure(2)
a_vals = np.linspace(-2, 10, 1000)
b_vals = dec_bound_func(a_vals)
plt.plot(a_vals, b_vals)
plt.plot(data_class1[:, 0], data_class1[:, 1], '+r', label='Clase 1')
plt.plot(data_class2[:, 0], data_class2[:, 1], '*b', label='Clase 2')
plt.axvline(0, color='black')
plt.axhline(0, color='black')
plt.xlim([-2, 10])
plt.ylim([-2.5, 10])
plt.xlabel("x1")  # Como en el original (posible error de labeling)
plt.ylabel("x2")
plt.grid(True)
plt.show()


#%%

import sympy as sp
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")  # Suprime advertencia de cov no simétrica

# Definir símbolos
a, b = sp.symbols('a b')
x = sp.Matrix([a, b])

# Tus matrices
mu_1 = sp.Matrix([3, 4])
mu_2 = sp.Matrix([3, -2])
sigma_1 = sp.Matrix([[sp.Rational(1,2), 1], [sp.Rational(-1,2), 2]])  # No simétrica; considera [[1/2, -1/2], [-1/2, 2]] si era error
sigma_2 = sp.Matrix([[2, 1], [1, 2]])

# Inversas simbólicas
inv_sig1 = sigma_1.inv()
inv_sig2 = sigma_2.inv()

pclass1 = sp.Rational(1,2)
pclass2 = sp.Rational(1,2)

# Componentes para sg1
W1 = -sp.Rational(1,2) * inv_sig1
w1 = inv_sig1 * mu_1
w10 = -sp.Rational(1,2) * (mu_1.T * inv_sig1 * mu_1)[0] - sp.Rational(1,2) * sp.log(sigma_1.det()) + sp.log(pclass1)

# Componentes para sg2
W2 = -sp.Rational(1,2) * inv_sig2
w2 = inv_sig2 * mu_2
w20 = -sp.Rational(1,2) * (mu_2.T * inv_sig2 * mu_2)[0] - sp.Rational(1,2) * sp.log(sigma_2.det()) + sp.log(pclass2)

# Expresiones discriminantes escalares
quad1 = (x.T * W1 * x)[0]
lin1 = (w1.T * x)[0]
sg1_scalar = quad1 + lin1 + w10

quad2 = (x.T * W2 * x)[0]
lin2 = (w2.T * x)[0]
sg2_scalar = quad2 + lin2 + w20

sg = sg1_scalar - sg2_scalar

# Frontera de decisión: resolver sg = 0 para b en términos de a (dos soluciones)
dec_bound = sp.solve(sg, b)

# Lambdify para ambas ramas
sg1_func = sp.lambdify((a, b), sg1_scalar, 'numpy')
sg2_func = sp.lambdify((a, b), sg2_scalar, 'numpy')
dec_bound_func1 = sp.lambdify(a, dec_bound[0], 'numpy')  # Rama inferior (menos √)
dec_bound_func2 = sp.lambdify(a, dec_bound[1], 'numpy')  # Rama superior (más √)

# Imprimir ecuación (elige la rama superior para la separación principal)
print("Ecuación de la frontera (rama superior, x2 en términos de x1):")
print(sp.simplify(dec_bound[1]))
print("\nDiscriminante D (siempre >0): 9x1² - 24x1 - 3ln(16) + 792")

# Gráfica 1: Frontera de decisión (ambas ramas)
plt.figure(1)
a_vals = np.linspace(0, 6, 1000)
b_vals1 = dec_bound_func1(a_vals)
b_vals2 = dec_bound_func2(a_vals)

# Graficar ambas
plt.plot(a_vals, b_vals1, 'k--', alpha=0.5, label='Rama inferior')  # Punteada, semi-transparente
plt.plot(a_vals, b_vals2, 'k-', linewidth=2, label='Rama superior')  # Sólida, principal

plt.axvline(0, color='black', linestyle='--', alpha=0.5)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.xlim([0, 6])
plt.ylim([-30, 6])  # Ampliado para ver ambas ramas; cambia a [-4,6] si solo quieres la superior
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Decision Boundary")
plt.legend()
plt.grid(True)
plt.show()

# Generar datos aleatorios
n = 100
mu_1_np = np.array([3., 4.])
mu_2_np = np.array([3., -2.])
sigma_1_np = np.array([[0.5, 1.], [-0.5, 2.]])
sigma_2_np = np.array([[2., 1.], [1., 2.]])
data_class1 = multivariate_normal.rvs(mean=mu_1_np, cov=sigma_1_np, size=n)
data_class2 = multivariate_normal.rvs(mean=mu_2_np, cov=sigma_2_np, size=n)

# Clasificación real
all_points = np.vstack([data_class1, data_class2])
true_labels = np.hstack([np.ones(n), np.zeros(n)])
predicted_labels = np.zeros(2*n)

for i in range(2*n):
    pt = all_points[i]
    g1_pt = sg1_func(pt[0], pt[1])
    g2_pt = sg2_func(pt[0], pt[1])
    if g1_pt > g2_pt:
        predicted_labels[i] = 1
    else:
        predicted_labels[i] = 0

class1_correct = np.sum((true_labels == 1) & (predicted_labels == 1))
class2_correct = np.sum((true_labels == 0) & (predicted_labels == 0))
accuracy_class1 = (class1_correct / n) * 100
accuracy_class2 = (class2_correct / n) * 100
overall_accuracy = ((class1_correct + class2_correct) / (2*n)) * 100

print(f"\nPrecisión clase 1: {accuracy_class1:.2f}%")
print(f"Precisión clase 2: {accuracy_class2:.2f}%")
print(f"Precisión general: {overall_accuracy:.2f}%")

# Gráfica 2: Frontera + puntos (ambas ramas)
plt.figure(2)
#plt.plot(a_vals, b_vals1, 'k--', alpha=0.5 label='Rama inferior')
plt.plot(a_vals, b_vals2, 'k-', linewidth=2)#, label='Rama superior')
plt.plot(data_class1[:, 0], data_class1[:, 1], '+r')
plt.plot(data_class2[:, 0], data_class2[:, 1], '*b')
plt.axvline(0, color='black', linestyle='--', alpha=0.5)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.xlim([0, 6])
plt.ylim([-4, 6])  # Aquí solo la superior; ajusta si quieres ver la inferior
plt.xlabel("x1")
plt.ylabel("x2")
#plt.title("Decision Boundary + Datos")
plt.legend()
plt.grid(True)
plt.show()
