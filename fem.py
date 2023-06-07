#!/usr/bin/python3

"""
Author:
Date:
Brief: Simple linear finite element model for a bending beam with changing cross-section.
"""

import numpy as np

class FiniteElementModel:
    """ Finite Element Model for Linear Structural Analysis of a Rotor Blade.

    This class handles the the finite element model for a linear structural
    analysis of a rotor blade under a static load. The beam is analyzed as
    a slender beam with a discrete number of elements. The coordinate frame
    of the beam is shown below for a 4 element model. 

      Z
      ^
      |
      |                 F_{z_2}  F_{z_3}
    <-+-\        F_{z_1}   ^        ^    F_{z_4}
      | |  F_{z_0} ^       |        |       ^
     ---/  ^       |       |        |       |
      |    +-------+-------+--------+-------+
      +----+ Elm 0 | Elm 1 | Elm 2  | Elm 3 | --> X
      |    +-------+-------+--------+-------+
    

    The blade has a span in the X-axis direction and revolves around the
    vertical Z-axis. For a slender beam, we constrain the the movements
    of the element interfaces to only the YZ-plane and only rotations
    about the X-axis to achieve only a 6-DOF per element. Additionaly, we
    assume that the beam is made of a uniform, isotropic material throughout
    the blade and that the cross-section of the blade is symmetric airfoils.

    Class Attributes:
        n_elements (int) : number of elements for the blade.
        n_nodes (int) : number of nodes in graph.
        E (float) : Material Young's Modulus of Elasticity.
        G (float) : Material Shear Modulus.
        Iyy (np.ndarray) : Y-axis bending moment of inertia of symmetric airfoil.
        Izz (np.ndarray) : Z-axis bending moment of inertia of symmetric airfoil.
        J (np.ndarray) : X-axis rotational moment of inertia of symmetric airfoil. 

    """

    @property
    def Iyy(self):
        return self.__Iyy

    @Iyy.setter
    def Iyy(self, val):
        assert isinstance(val, np.ndarray), "Y-axis bend inertia must be a vector of elements"
        assert val.size == self.n_elements, (
            "Y-axis bending inertia must be the same size as the number of elements."
        )
        self.__Iyy = val

    @property
    def Izz(self):
        return self.__Izz

    @Iyy.setter
    def Izz(self, val):
        assert isinstance(val, np.ndarray), "Z-axis bend inertia must be a vector of elements"
        assert val.size == self.n_elements, (
            "Z-axis bending inertia must be the same size as the number of elements."
        )
        self.__Izz = val

    @property
    def J(self):
        return self.__J
        
    @J.setter
    def J(self, val):
        assert isinstance(val, np.ndarray), (
            "X-axis rotational inertia must be a vector of elements"
        )
        assert val.size == self.n_elements, (
            "X-axis roational inertia must be the same size as the number of elements."
        )
        self.__J = val

    @property
    def theta(self):
        return self.__theta
        
    @theta.setter
    def theta(self, val):
        assert isinstance(val, np.ndarray), (
            "X-axis rotational angles must be a vector of elements"
        )
        assert val.size == self.n_elements, (
            "X-axis rotational angles must be the same size as the number of elements."
        )
        self.__theta = val
        
    def __init__(self, n_elements, E, G, rotor_length, Iyy=None, Izz=None, J=None, theta=None):
        """
        Args:
            n_elements: total number of elements
            E: young's modulus
            G: shear modulus
            rotor_length: length of the rotor blade

        """
        self.n_elements = n_elements
        self.n_nodes = n_elements + 1
        self.L = (rotor_length/n_elements)
        self.E = E
        self.G = G
        if Iyy is not None:
            self.Iyy = Iyy
        if Izz is not None:
            self.Izz = Izz
        if J is not None:
            self.J = J
        if theta is not None:
            self.theta = theta
        #self.K_inv=np.zeros(((self.n_nodes-1) * 5, (self.n_nodes-1) * 5))

    def change_iteration(self, Iy, Iz, J, theta):
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.theta = theta
        
    def principle_element_stiffness_matrix(self, element):
        Iy = self.Iyy[element]
        Iz = self.Izz[element]
        J = self.J[element]
        L = self.L
        E = self.E
        G = self.G      

        K_local = np.array(
            [[ 12*Iz  ,  0     , 0        , 0         , 6*Iz*L  ,-12*Iz   ,  0     ,0         , 0       , 6*Iz*L  ],
             [  0     , 12*Iy  , 0        ,-6*Iy*L    , 0       ,  0      ,-12*Iy  ,0         ,-6*Iy*L  , 0       ],
             [  0     ,  0     , G*J*L*L/E, 0         , 0       ,  0      ,  0     ,-G*J*L*L/E, 0       , 0       ],
             [  0     , -6*Iy*L, 0        , 4*E*Iy*L*L, 0       ,  0      , 6*Iy*L ,0         , 2*Iy*L*L, 0       ],
             [  6*Iz*L,  0     , 0        , 0         , 4*Iz*L*L, -6*Iz*L ,  0     ,0         , 0       , 2*Iz*L*L],
             [-12*Iz  ,  0     , 0        , 0         ,-6*Iz*L  , 12*Iz   ,  0     ,0         , 0       ,-6*Iz*L ],
             [  0     ,-12*Iy  , 0        , 6*Iy*L    , 0       ,  0      , 12*Iy  ,0         , 6*Iy*L  , 0       ],
             [  0     ,  0     ,-G*J*L*L/E, 0         , 0       ,  0      ,  0     ,G*J*L*L/E , 0       , 0       ],
             [  0     , -6*Iy*L, 0        , 2*Iy*L*L  , 0       ,  0      ,  6*Iy*L,0         , 4*Iy*L*L, 0       ],
             [  6*Iz*L,  0     , 0        , 0         , 2*Iz*L*L, -6*Iz*L ,  0     ,0         , 0       , 4*Iz*L*L]],
            dtype='f'
        )
        K_local *= E/(L**3)
        return K_local
        
    def element_stiffness_matrix(self, element):
        K_local = self.principle_element_stiffness_matrix(element)
        theta = self.theta[element]
        R_0 = np.array(
            [[np.cos(theta),-np.sin(theta), 0 , 0            , 0            ],
             [np.sin(theta), np.cos(theta), 0 , 0            , 0            ],
             [0            , 0            , 1 , 0            , 0            ],
             [0            , 0            , 0 , np.cos(theta),-np.sin(theta)],
             [0            , 0            , 0 , np.sin(theta), np.cos(theta)]],
            dtype='f'
        )
        R = np.zeros((10,10),dtype='f')
        R[:5,:5] = R_0
        R[5:10,5:10] = R_0
        # K=np.matmul(np.matmul(np.transpose(R),K_local,),R)
        K_global_frame = R.T @ K_local @ R  # Rotate local stiffness matrix into global frame
        return K_global_frame

    def global_stiffness_matrix(self):
        K = np.zeros((self.n_nodes * 5, self.n_nodes * 5))
        for element in range(self.n_elements):
            K_element = self.element_stiffness_matrix(element)
            dofs = [element*5 ,element*5+1 ,element*5+2 ,element*5+3 ,element*5+4 ,element*5+5 ,element*5+6 ,element*5+7 ,element*5+8 ,element*5+9]
            for i, dof_i in enumerate(dofs):
                for j, dof_j in enumerate(dofs):
                    K[dof_i, dof_j] += K_element[i, j]
        return K
    
    def solve_displacement(self, f):
        """ Solves the structural displacement for a given force.

        Args:
            f (np.ndarray) : force and moments at each element
                           : f = [..., [f_y, f_z, m_x, m_y, m_z]_i, ...]
        
        Returns:
            u (np.ndarray) : displacements of each element interface
                           : u = [..., [u_y, u_z, phi_x, phi_y, phi_z]_i, ...]
        """
        
        # Apply boundary conditions (fixed at left end)
        bc_dofs = [0, 1, 2, 3, 4]  # Degrees of freedom to fix (in increasing order)
        K_solve=self.global_stiffness_matrix()
        for i, dof in enumerate(bc_dofs): # Get the K matrix to solve for u
            K_solve = np.delete(K_solve, dof - i, axis=0)
            K_solve = np.delete(K_solve, dof - i, axis=1)

        # Preallocate solution
        u = np.zeros((self.n_nodes - 1)*5)
        
        # Solve for displacement
        u = np.linalg.solve(K_solve, f)
        # u = np.concatenate((np.array([[0],[0],[0],[0],[0]]),u))
        u = np.concatenate((np.zeros((len(bc_dofs), 1)), u))
        #F=np.matmul(self.global_stiffness_matrix(),u)
        return u
    
    def K_inverse(self):
        # Apply boundary conditions (fixed at left end)
        bc_dofs = [0, 1, 2, 3, 4]  # Degrees of freedom to fix (in increasing order)
        K = self.global_stiffness_matrix()
        K = np.delete(K, bc_dofs, axis=0)  # Remove force calculations at fixed nodes
        K = np.delete(K, bc_dofs, axis=1)  # Remove zero displacement contributions
        K_inv = np.linalg.inv(K)
        return K_inv

    
    def element_stresses(self, u):
        """ Given a displacement of the nodes, return the element stresses

        This is using Bernoulli beam element theory with some additional assumptions to make
        the 

        Ref: 
            [1] https://www.sesamx.io/blog/beam_finite_element/
        """
        pass
