import numpy as np
import fem_functions
import optimizer
import matplotlib.pyplot as plt


n_elements=2
E=7.31*(10**10)
G=2.8*(10**10)
sigma_v=3.24*(10**6)
u_max=np.array([[0.1],[0.1],[0.1],[0.1],[0.1]])
f=np.array([[10],[10],[10],[10],[10],[10],[10],[10],[10],[10]])

rotor_length=1
tr=0.15
tt=0.1
cr=1
ct=0.5
theta_r=0.01
theta_t=0.05

x0=np.array([tr,tt,cr,ct,theta_r,theta_t])
w=np.array([1,1,1,1,1,1])

fem_obj=lambda y: fem_functions.F_objective(y,w)
fem_obj=optimizer.numerical_gradient(fem_obj)
fem_c=lambda y: -1*fem_functions.F_constraints(n_elements,E,G,sigma_v,u_max,f,rotor_length,y)
fem_c=optimizer.numerical_gradient(fem_c)
eta=0.01
opt_grad = optimizer.GradientOptimizer(fem_obj, eta, [fem_c])

result = opt_grad.optimize(x0, max_iter=200)
xh = result['xh']
Tr=xh[:,1]

fig, ax = plt.subplots(6,1)
ax[0].plot(xh[:,0], linewidth=2.)
ax[1].plot(xh[:,1], linewidth=2.)
ax[2].plot(xh[:,2], linewidth=2.)
ax[3].plot(xh[:,3], linewidth=2.)
ax[4].plot(xh[:,4], linewidth=2.)
ax[5].plot(xh[:,5], linewidth=2.)
plt.show()