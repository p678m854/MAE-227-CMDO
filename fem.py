#!/usr/bin/python3

"""
Author:
Date:
Brief: Simple linear finite element model for a bending beam with changing cross-section.
"""

import numpy as np
from optimizer import numerical_gradient

def max_shear_stress(Torque, Iyy, D, A, C, r):
    return (
        Torque*D/(1 + np.pi*np.pi*np.power(D,4.)/(16.*A*A))
        *(0.25/Iyy + 4./(A*C))
        *(1. + 0.15*(np.pi*np.pi*D*D*D*D/(16.*A*A) - D/(2.*r)))
    )

def RotationMatrix(theta):
    return np.array(
        [[np.cos(theta),-np.sin(theta), 0 , 0            , 0            ],
         [np.sin(theta), np.cos(theta), 0 , 0            , 0            ],
         [0            , 0            , 1 , 0            , 0            ],
         [0            , 0            , 0 , np.cos(theta),-np.sin(theta)],
         [0            , 0            , 0 , np.sin(theta), np.cos(theta)]],
        dtype='f'
    )

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

    class Element:
        """ Handles information about the elements """

        def __init__(self, id_num, owner):
            self.id_num = id_num
            self.owner = owner

            self.Iyy = 0.
            self.Izz = 0.
            self.J = 0.
            self.A = 0.
            self.D = 0.
            self.r = 0.
            self.theta = 0.
            self.yp = 0.
            self.zp = 0.

        @property
        def Iyy(self):
            return self.__Iyy

        @property
        def Izz(self):
            return self.__Izz

        @property
        def J(self):
            return self.__J

        @property
        def A(self):
            return self.__A

        @property
        def D(self):
            return self.__D

        @property
        def r(self):
            return self.__r

        @property
        def theta(self):
            return self.__theta

        @Iyy.setter
        def Iyy(self, val):
            self.__Iyy = val

        @Izz.setter
        def Izz(self, val):
            self.__Izz = val

        @J.setter
        def J(self, val):
            self.__J = val

        @A.setter
        def A(self, val):
            self.__A = val

        @D.setter
        def D(self, val):
            self.__D = val

        @r.setter
        def r(self, val):
            self.__r = val

        @theta.setter
        def theta(self, val):
            self.__theta = val

        @property
        def yp(self):
            return self.__yp

        @yp.setter
        def yp(self, val):
            self.__yp = val

        @property
        def zp(self):
            return self.__zp

        @zp.setter
        def zp(self, val):
            self.__zp = val

        @property
        def chord(self):
            return self.__chord

        @chord.setter
        def chord(self, val):
            self.__chord = val

        def principle_element_stiffness_matrix(self):
            return self.owner.principle_element_stiffnes_matrix(self.id_num)

        def element_stiffness_matrix(self):
            return self.owner.element_stiffness_matrix(self.id_num)

        def max_axial_stress_magnitude(self, My, Mz):
            return np.max(np.abs(My/self.Iyy*self.zp - Mz/self.Izz*self.yp))

        def max_shear_stress(self, Mx):
            return max_shear_stress(Mx, self.Iyy, self.D, self.A, self.chord, self.r)

        def determine_forces(self, u1, u2):
            """ Edge displacements u1 and u2 into forces and moments """

            u_vec = np.hstack((u1, u2))
            K = self.element_stiffness_matrix()
            f_vec = K @ u_vec
            f_vec = f_vec.reshape((2, f_vec.size//2))[0]

            return f_vec

        def vonMises_stress(self, f_vec):
            
            tau_max = self.max_shear_stress(f_vec[2])
            sigma_max_abs = self.max_axial_stress_magnitude(*f_vec[3:])

            return sigma_max_abs*sigma_max_abs + 3.*tau_max*tau_max

        def displacement_to_vonMises_stress(self, u1, u2):
            return self.vonMises_stress(self.determine_forces(u1, u2))

        
            
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
        self.R_vec = np.arange(self.n_elements)*self.L
        self.E = E
        self.G = G
        if Iyy is not None:
            self.Iyy = Iyy
        else:
            self.__Iyy = None
        if Izz is not None:
            self.Izz = Izz
        else:
            self.__Izz = None
        if J is not None:
            self.J = J
        if theta is not None:
            self.theta = theta
        #self.K_inv=np.zeros(((self.n_nodes-1) * 5, (self.n_nodes-1) * 5))

        # This is what we will use to 
        self._element_list = [self.Element(i, self) for i in range(self.n_elements)]

    def apply_profiles(self):

        struct_parameters = [
            'Iyy', 'Izz', 'J',
            'A', 'D', 'r', 'yp', 'zp',
            'theta', 'chord'
        ]
        for prop in struct_parameters:
            prop_func = '{}_func'.format(prop)
            if prop_func in self.profiles.keys():
                
                func = self.profiles[prop_func]

                prop_vec = func(self.R_vec)
                setattr(self, prop, prop_vec)
            
                for i, elm in enumerate(self._element_list):
                    setattr(elm, prop, prop_vec[i])
        
    def change_iteration(self, Iy, Iz, J, theta, f):
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.theta = theta
        self.f = f
        
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


    def translate_aerodynamic_forces_and_moments_to_structural(self, F_aero):
        """ Shifting a half element down and adding free pressure boundary """
        F = F_aero.copy()
        F[-2] = -F[2]*self.L/2.  # Added My term
        F[-1] = F[1]*self.L/2.  # Added Mz term
        F = np.vstack((F, np.zeros((1,) + F.shape[1:])))
        return F[1:]
    
    def critical_vonMises_stress(self, f):
        """ Returns the squared stress """
        u = self.solve_displacement(f)
        u = u.reshape((self.n_nodes, u.size//self.n_nodes))

        stresses = np.zeros(self.n_elements)
        for elm in self._element_list:
            stresses[elm.id_num] = (
                elm.displacement_to_vonMises_stress(u[elm.id_num], u[elm.id_num + 1])
            )

        return np.max(stresses)
