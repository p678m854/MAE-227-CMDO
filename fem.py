import numpy as np

class finite_element_model:
    Iy=None
    Iz=None
    J=None
    theta=None
    f=None


    def __init__(self,n_elements,E,G,rotor_length):
        """
        Args:
            n_elements: total number of elements
            E: young's modulus
            G: shear modulus
            rotor_length: length of the rotor blade

        """
        """
        Vars:
            Iy: numpy array of the Iy of beam elements
            Iz: numpy array of the Iz of beam elements
            J: numpy array of the J of beam elements            
            theta: numpy array of theta of beam elements
            f: force vector as forces acting at each node [[fy],[fz],[mx],[my],[mz]]
            
        """
        self.n_elements=n_elements
        self.n_nodes=n_elements + 1
        self.L=(rotor_length/n_elements)
        self.E=E
        self.G=G
        #self.K_inv=np.zeros(((self.n_nodes-1) * 5, (self.n_nodes-1) * 5))

    def change_iteration(self,Iy,Iz,J,theta,f):
        finite_element_model.Iy=Iy
        finite_element_model.Iz=Iz
        finite_element_model.J=J
        finite_element_model.theta=theta
        finite_element_model.f=f
        

    def element_stiffness_matrix(self,element):
        Iy=finite_element_model.Iy[element]
        Iz=finite_element_model.Iz[element]
        theta=finite_element_model.theta[element]
        J=finite_element_model.J[element]
        L=self.L
        E=self.E
        G=self.G      

        K_local = np.array([[12*Iz  ,0     ,0         ,0         ,6*Iz*L  ,-12*Iz ,0      ,0         ,0       ,6*Iz*L  ],
                            [0      ,12*Iy ,0         ,-6*Iy*L    ,0       ,0      ,-12*Iy ,0         ,-6*Iy*L,0       ],
                            [0      ,0     ,G*J*L*L/E ,0         ,0       ,0      ,0      ,-G*J*L*L/E,0       ,0       ],
                            [0      ,-6*Iy*L,0         ,4*E*Iy*L*L,0       ,0      ,6*Iy*L,0         ,2*Iy*L*L,0       ],
                            [6*Iz*L ,0     ,0         ,0         ,4*Iz*L*L,-6*Iz*L,0      ,0         ,0       ,2*Iz*L*L],
                            [-12*Iz ,0     ,0         ,0         ,-6*Iz*L ,12*Iz  ,0      ,0         ,0       ,-6*Iz*L ],
                            [0      ,-12*Iy,0         ,6*Iy*L   ,0       ,0      ,12*Iy  ,0         ,6*Iy*L ,0       ],
                            [0      ,0     ,-G*J*L*L/E,0         ,0       ,0      ,0      ,G*J*L*L/E ,0       ,0       ],
                            [0      ,-6*Iy*L,0         ,2*Iy*L*L  ,0       ,0      ,6*Iy*L,0         ,4*Iy*L*L,0       ],
                            [6*Iz*L ,0     ,0         ,0         ,2*Iz*L*L,-6*Iz*L,0      ,0         ,0       ,4*Iz*L*L]],dtype='f')
        K_local *= E / L ** 3
        R_0=np.array([[np.cos(theta) ,-np.sin(theta),0 ,0             ,0            ],
                    [np.sin(theta),np.cos(theta),0 ,0             ,0            ],
                    [0             ,0            ,1 ,0             ,0            ],
                    [0             ,0            ,0 ,np.cos(theta) ,-np.sin(theta)],
                    [0             ,0            ,0 ,np.sin(theta),np.cos(theta)]],dtype='f')
        R=np.zeros([10,10],dtype='f')
        R[0:5,0:5]=R_0
        R[5:10,5:10]=R_0
        K=np.matmul(np.matmul(np.transpose(R),K_local,),R)
        return K
    

    def principle_element_stiffness_matrix(self,element):
        Iy=finite_element_model.Iy[element]
        Iz=finite_element_model.Iz[element]
        J=finite_element_model.J[element]
        L=self.L
        E=self.E
        G=self.G      

        K_local = np.array([[12*Iz  ,0     ,0         ,0         ,6*Iz*L  ,-12*Iz ,0      ,0         ,0       ,6*Iz*L  ],
                            [0      ,12*Iy ,0         ,-6*Iy*L    ,0       ,0      ,-12*Iy ,0         ,-6*Iy*L,0       ],
                            [0      ,0     ,G*J*L*L/E ,0         ,0       ,0      ,0      ,-G*J*L*L/E,0       ,0       ],
                            [0      ,-6*Iy*L,0         ,4*E*Iy*L*L,0       ,0      ,6*Iy*L,0         ,2*Iy*L*L,0       ],
                            [6*Iz*L ,0     ,0         ,0         ,4*Iz*L*L,-6*Iz*L,0      ,0         ,0       ,2*Iz*L*L],
                            [-12*Iz ,0     ,0         ,0         ,-6*Iz*L ,12*Iz  ,0      ,0         ,0       ,-6*Iz*L ],
                            [0      ,-12*Iy,0         ,6*Iy*L   ,0       ,0      ,12*Iy  ,0         ,6*Iy*L ,0       ],
                            [0      ,0     ,-G*J*L*L/E,0         ,0       ,0      ,0      ,G*J*L*L/E ,0       ,0       ],
                            [0      ,-6*Iy*L,0         ,2*Iy*L*L  ,0       ,0      ,6*Iy*L,0         ,4*Iy*L*L,0       ],
                            [6*Iz*L ,0     ,0         ,0         ,2*Iz*L*L,-6*Iz*L,0      ,0         ,0       ,4*Iz*L*L]],dtype='f')
        K_local *= E / L ** 3
        return K_local
    

    def global_stiffness_matrix(self):
        K = np.zeros((self.n_nodes * 5, self.n_nodes * 5))
        for element in range(self.n_elements):
            K_element = self.element_stiffness_matrix(element)
            dofs = [element*5 ,element*5+1 ,element*5+2 ,element*5+3 ,element*5+4 ,element*5+5 ,element*5+6 ,element*5+7 ,element*5+8 ,element*5+9]
            for i, dof_i in enumerate(dofs):
                for j, dof_j in enumerate(dofs):
                    K[dof_i, dof_j] += K_element[i, j]
        return K
    
    def solve_displacement(self):
        # Apply boundary conditions (fixed at left end)
        bc_dofs = [0, 1, 2, 3, 4]  # Degrees of freedom to fix (in increasing order)
        K_solve=self.global_stiffness_matrix()
        for i, dof in enumerate(bc_dofs): # Get the K matrix to solve for u
            K_solve = np.delete(K_solve, dof - i, axis=0)
            K_solve = np.delete(K_solve, dof - i, axis=1)
        u = np.zeros((self.n_nodes-1)* 5)
        
        # Solve for displacement
        f =self.f
        u = np.linalg.solve(K_solve, f)
        u = np.concatenate((np.array([[0],[0],[0],[0],[0]]),u))
        #F=np.matmul(self.global_stiffness_matrix(),u)
        return u
    
    def K_inverse(self):
        # Apply boundary conditions (fixed at left end)
        bc_dofs = [0, 1, 2, 3, 4]  # Degrees of freedom to fix (in increasing order)
        K=self.global_stiffness_matrix()
        for i, dof in enumerate(bc_dofs): # Get the K~ matrix to invert
            K = np.delete(K, dof - i, axis=0)
            K = np.delete(K, dof - i, axis=1)
        K_inv=np.linalg.inv(K)
        return K_inv
        

