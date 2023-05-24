import numpy as np
from airfoils import NACA4Series as airfoil
from fem import finite_element_model as FEM
from fem_constraints import constraints as constraints


n_elements=2
Ix=np.array([[1],[1]])
Iy=np.array([[1],[1]])
Iz=np.array([[1],[1]])
J=np.array([[1],[1]])
E=1
G=1
theta=np.array([[0],[0]])
f=np.array([[0],[0],[0],[0],[0],[1],[1],[0],[0],[0]])
rotor_length=1
sigma_v=1
u_max=np.array([[0.1],[0.1],[0.1],[0.1],[0.1]])

D=np.array([[1],[1]])
Area=np.array([[1],[1]])
C=np.array([[1],[1]])
r=np.array([[1],[1]])

yp=np.array([[1,1],[1,1]])
zp=np.array([[1,1],[1,1]])


a=FEM(n_elements,E,G,rotor_length)
a.change_iteration(Iy,Iz,J,theta,f)
#print(a.K_inv)
a.K_inv=a.K_inverse()
print(a.K_inv)

b=constraints(n_elements,sigma_v,u_max)

b.change_iteration(f,K_inv=a.K_inverse())
e=0
b.change_element(e,Ix[e],Iy[e],Iz[e],D[e],Area[e],C[e],r[e],theta[e],a.principle_element_stiffness_matrix(e))
p=0
b.change_point(yp[e,p],zp[e,p])

print(b.von_mises())
print(b.strain())