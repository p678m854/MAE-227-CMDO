import numpy as np
from airfoils import NACA4Series as airfoil
from fem import finite_element_model as FEM
from fem_constraints import constraints as constraints


def F_constraints(n_elements,E,G,sigma_v,u_max,f,rotor_length,x):
    tr=x[0]
    tt=x[1]
    cr=x[2]
    ct=x[3]
    theta_r=x[4]
    theta_t=x[5]
    t=np.linspace(tr,tt,n_elements)
    c=np.linspace(cr,ct,n_elements)
    theta=np.linspace(theta_r,theta_t,n_elements)


    a=FEM(n_elements,E,G,rotor_length)
    b=constraints(n_elements,sigma_v,u_max)

    Iz=np.zeros(n_elements)
    Iy=np.zeros(n_elements)
    Ix=np.zeros(n_elements)
    J=np.zeros(n_elements)
    D=np.zeros(n_elements)
    Area=np.zeros(n_elements)
    r=np.zeros(n_elements)
    yp=np.zeros((n_elements,4))
    zp=np.zeros((n_elements,4))

    min_eig=10**10
    for i in range(n_elements):
        af=airfoil(t[i])
        Iz[i]=af.Ixx(c[i])
        Iy[i]=af.Iyy(c[i])
        Ix[i]=Iz[i]+Iy[i] #from perpendicular axis theorem
        J[i]=af.J(c[i])
        D[i]=t[i]
        Area[i]=af.A(c[i])
        yp[i,:]=np.array([0,t[i]/2,0,-t[i]/2])
        LeadingE_z=af.center_x(c[i])
        LaggingE_z=-c[i]+LeadingE_z
        Maxt_z=af.center_maxt(c[i])
        r[i]=af.curvature(Maxt_z,c[i])   
        zp[i,:]=np.array([LeadingE_z,Maxt_z,LaggingE_z,Maxt_z]) 
    
    a.change_iteration(Iy,Iz,J,theta,f)
    a.K_inv=a.K_inverse()
    b.change_iteration(f,K_inv=a.K_inverse())


    for i in range(n_elements): 
        b.change_element(i,Ix[i],Iy[i],Iz[i],D[i],Area[i],c[i],r[i],theta[i],a.principle_element_stiffness_matrix(i))
        for j in range(4):
            b.change_point(yp[i,j],zp[i,j])
            min_eig=min(min(np.linalg.eigvals(b.von_mises_lmi())),min_eig)
            min_eig=min(min(np.linalg.eigvals(b.strain_lmi())),min_eig)
    return min_eig
    
def F_objective(x,w):
    f=np.matmul(np.transpose(x),w)
    return f