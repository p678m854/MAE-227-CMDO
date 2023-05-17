import numpy as np
import matplotlib.pyplot as plt

def finite_element_model():
    # Rotor parameters
    L_rotor = 1.0 # Length of rotor (m)
    A = 5e-3 # Cross-sectional area of rotor (m^2)
    Iy = 4e-11 # Second moment of area of the rotor for bending in xy plane (m^4)
    Iz = 4e-11 # Second moment of area of the rotor for bending in zx plane (m^4)
    J = 4e-11 # Polar moment of area of the rotor for twisting about x axis (m^4)
    E = 2e11 # Young's modulus of beam (Pa)
    G = 2e10 # Shear modulus of beam (Pa)
    nu = 0.3 # Poisson's ratio of rotor

    # Finite element mesh
    n_elements = 3
    n_nodes = n_elements + 1
    L = L_rotor/n_elements
    x = np.linspace(0, L, n_nodes)
    y = np.zeros(n_nodes)
    element_nodes = np.array([(i, i+1) for i in range(n_elements)])

    # Element stiffness matrix
    def element_stiffness_matrix(E, Iy, Iz, J, L, theta):
        K_local = np.array([[12*Iy  ,0     ,0         ,0         ,6*Iy*L  ,-12*Iy ,0      ,0         ,0       ,6*Iy*L  ],
                            [0      ,12*Iz ,0         ,6*Iz*L    ,0       ,0      ,-12*Iz ,0         ,6*Iz*L*L,0       ],
                            [0      ,0     ,G*J*L*L/E ,0         ,0       ,0      ,0      ,-G*J*L*L/E,0       ,0       ],
                            [0      ,6*Iz*L,0         ,4*E*Iz*L*L,0       ,0      ,-6*Iz*L,0         ,2*Iz*L*L,0       ],
                            [6*Iy*L ,0     ,0         ,0         ,4*Iy*L*L,-6*Iy*L,0      ,0         ,0       ,2*Iy*L*L],
                            [-12*Iy ,0     ,0         ,0         ,-6*Iy*L ,12*Iy  ,0      ,0         ,0       ,-6*Iy*L ],
                            [0      ,-12*Iz,0         ,-6*Iz*L   ,0       ,0      ,12*Iz  ,0         ,-6*Iz*L ,0       ],
                            [0      ,0     ,-G*J*L*L/E,0         ,0       ,0      ,0      ,G*J*L*L/E ,0       ,0       ],
                            [0      ,6*Iz*L,0         ,2*Iz*L*L  ,0       ,0      ,-6*Iz*L,0         ,4*Iz*L*L,0       ],
                            [6*Iy*L ,0     ,0         ,0         ,2*Iy*L*L,-6*Iy*L,0      ,0         ,0       ,4*Iy*L*L]])
        K_local *= E / L ** 3
        R_0=np.array([[np.cos(theta) ,-np.sin(theta),0 ,0             ,0            ],
                    [np.sin(theta),np.cos(theta),0 ,0             ,0            ],
                    [0             ,0            ,1 ,0             ,0            ],
                    [0             ,0            ,0 ,np.cos(theta) ,-np.sin(theta)],
                    [0             ,0            ,0 ,np.sin(theta),np.cos(theta)]])
        R=np.zeros([10,10])
        R[0:5,0:5]=R_0
        R[5:10,5:10]=R_0
        K=np.matmul(R,np.matmul(K_local,np.transpose(R)))
        return K

    # Global stiffness matrix
    K = np.zeros((n_nodes * 5, n_nodes * 5))
    for element in range(n_elements):
        K_element = element_stiffness_matrix(E, Iy, Iz, J, L, 0.1)
        dofs = [element*5 ,element*5+1 ,element*5+2 ,element*5+3 ,element*5+4 ,element*5+5 ,element*5+6 ,element*5+7 ,element*5+8 ,element*5+9]
        for i, dof_i in enumerate(dofs):
            for j, dof_j in enumerate(dofs):
                K[dof_i, dof_j] += K_element[i, j]

    # Apply boundary conditions (fixed at left end)
    bc_dofs = [0, 1, 2, 3, 4]  # Degrees of freedom to fix (in increasing order)
    K_solve=K
    for i, dof in enumerate(bc_dofs): # Get the K matrix to solve for u
        K_solve = np.delete(K_solve, dof - i, axis=0)
        K_solve = np.delete(K_solve, dof - i, axis=1)
    u = np.zeros((n_nodes-1)* 5)

    # Solve for displacement
    f = np.ones((n_nodes-1)*5) # forces acting on the nodes
    u = np.linalg.solve(K_solve, f)
    u = np.concatenate((np.array([0,0,0,0,0]),u))
    return u