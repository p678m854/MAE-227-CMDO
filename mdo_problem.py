#!/usr/bin/python3

"""
Brief: Setup of the Multidisciplinary Design Optimization problem for the design
       of a helicopter rotor blade.
"""

from airfoils import NACA4Series
from bemt import BEMTAnalysis
import copy
from fem import FiniteElementModel
import matplotlib.pyplot as plt
import numpy as np
from optimizer import (
    isaffine, isaffineequality,
    add_gradient, numerical_gradient,
    GradientOptimizer, NesterovOptimizer
)
from scipy.interpolate import interp1d
from typing import Iterable
    

class LowerBound:
    def __init__(self, index, val):
        self.index = index
        self.val = val

    def __call__(self, x):
        return self.val - x[self.index]

    def gradient(self, x):
        grad = np.zeros(x.shape)
        grad[self.index] = -1.
        return grad

    @property
    def isaffine(self):
        return True

class UpperBound:
    def __init__(self, index, val):
        self.index = index
        self.val = val

    def __call__(self, x):
        return x[self.index] - self.val

    def gradient(self, x):
        grad = np.zeros(x.shape)
        grad[self.index] = 1.
        return grad

    @property
    def isaffine(self):
        return True

class SectionConstraint:

    def __init__(self, i_section, n_states_per_section, func):
        self.i_start = i_section*n_states_per_section
        self.i_stop = (i_section + 1)*n_states_per_section
        self.func = func
        
    def __call__(self, x):
        return self.func(x[self.i_start:self.i_stop])

    def gradient(self, x):
        grad = np.zeros(x.shape)
        grad[self.i_start:self.i_stop] = self.func.gradient(x[self.i_start:self.i_stop])
        return grad

    @property
    def isaffine(self):
        if hasattr(self.func, 'isaffine'):
            return self.func.isaffine
        else:
            return False
    

class RotorBladeAnalysis:
    """ Class to handle rotor blade analysis """

    # Valid parameters to use as state functions
    problem_params = [
        'chord',
        'thickness',
        'theta',
        'skin_thickness'
    ]
    n_states_per_section = len(problem_params)

    aero_parameters = [
        'sigma',
        'theta',
        'C_l_alpha',
        'C_d_0', 'd_1', 'd_2',
        'C_m_c4_alpha',
        'alpha_max'
    ]

    struct_parameters = [
        'Iyy', 'Izz', 'J',
        'A', 'D', 'r', 'yp', 'zp',
        'theta', 'chord'
    ]
    
    class BladeSection:
        """ Handles the cross-sections of the airfoil"""

        def __init__(self, airfoil, chord, theta, t_skin, Nb, R):

            # Get defining charateristics
            self.__airfoil = airfoil
            self.__chord = chord
            self.__theta = theta
            self.__t_skin = t_skin
            self.__Nb = Nb
            self.__R = R

            self.update_structural_resultant_properties()
            self.update_aerodynamic_resultant_properties()

        def update_structural_resultant_properties(self):
            """ These properties are a result of the defining properties """
            # There is coordinate shift here from airfoil to rotor frame
            self.Iyy = self.__airfoil.Ixx_skin(self.chord, self.skin_thickness)
            self.Izz = self.__airfoil.Iyy_skin(self.chord, self.skin_thickness)
            self.J = self.__airfoil.J_skin(self.chord, self.skin_thickness)
            self.A = self.__airfoil.A(self.chord)
            self.D = self.__airfoil.diameter_max_inscribed_circle(
                self.chord, self.__airfoil.t
            )
            # TODO: figure out this radius of curvature
            self.yp = np.array([
                self.__airfoil.center_x(self.chord),
                self.__airfoil.center_maxt(self.chord),
                self.__airfoil.center_x(self.chord) - self.chord,
                self.__airfoil.center_maxt(self.chord),
            ])
            self.zp = np.array([0., self.thickness, 0., -self.thickness])
            self.r = self.__airfoil.curvature(0.3, self.chord)

        def update_aerodynamic_resultant_properties(self):
            """ These properties are a result of the defining properties """
            self.C_l_alpha = self.__airfoil.c_l_alpha(self.__airfoil.t)
            self.C_d_0 = self.__airfoil.c_d_0(self.__airfoil.t)
            self.d_1 = self.__airfoil.d_1(self.__airfoil.t)
            self.d_2 = self.__airfoil.d_2(self.__airfoil.t)
            self.C_m_c4_alpha = self.__airfoil.c_m_c4_alpha(self.__airfoil.t)
            self.alpha_max = self.__airfoil.alpha_max(self.__airfoil.t)
            
        @property
        def airfoil(self):
            return str(self.__airfoil)

        @airfoil.setter
        def airfoil(self, val):
            raise ValueError("Cannot change airfoil series after its been selected")
    
        @property
        def chord(self):
            return self.__chord

        @chord.setter
        def chord(self, val):
            self.__chord = val
            self.update_structural_resultant_properties()

        @property
        def thickness(self):
            return self.chord*self.__airfoil.t

        @thickness.setter
        def thickness(self, val):
            self.__airfoil.t = val/self.chord
            self.update_structural_resultant_properties()
            self.update_aerodynamic_resultant_properties()

        @property
        def skin_thickness(self):
            return self.__t_skin

        @skin_thickness.setter
        def skin_thickness(self, val):
            self.__t_skin = val
            self.update_structural_resultant_properties()

        @property
        def sigma(self):
            """ Rotor solidity """
            return self.__Nb*self.chord/(np.pi*self.__R)
            
        @property
        def theta(self):
            return self.__theta

        @theta.setter
        def theta(self, val):
            self.__theta = val
    
    def __init__(
            self,
            Nb, R, r_profile_stations, n_elements_per_section,  # Blade parameters
            Omega, rho, # Aerodynamics parameter
            E, G,  # Structural Parameter
            aero_kargs=dict(),
            struct_kargs=dict(),
            **kargs
    ):
        """
        Args:
            Nb (int) : Number of helicopter blades
            R (float) : Radius of the blades
            r_profile_stations (np.ndarray) : nondimensional 
        """

        # Designer choices
        self.Nb = Nb
        self.R = R
        self.r_profile_stations = np.array(r_profile_stations)
        self.n_stations = len(r_profile_stations)

        # Analysis choices
        self.n_elements_per_section = n_elements_per_section
        self.n_elements = (self.n_stations - 1)*self.n_elements_per_section

        # Aerodynamic problem constraints
        self.rho = rho
        self.Omega = Omega
        
        # Structural problem constraints
        self.E = E  # Young's Modulus of Elasticity
        self.G = G  # Shear Modulus of Elasticity

        # Keyword arguments for subproblems
        self.aero_kargs = aero_kargs
        self.struct_kargs = struct_kargs

        # Save the subproblems
        self.bemt = BEMTAnalysis(
            n_elements=self.n_elements, Nb=self.Nb, r_hub=self.r_profile_stations[0],
            Omega=self.Omega, R=self.R, rho=self.rho,
            **self.aero_kargs
        )
        self.fem = FiniteElementModel(
            n_elements=self.n_elements, E=self.E, G=self.G, rotor_length=self.R
        )
        
        # Get arguments for constructing the sections
        section_chords = getattr(kargs, 'section_chords', 0.15*np.ones(self.n_stations))
        section_airfoils = getattr(kargs, 'section_airfoils', NACA4Series(0.12))
        section_t_skin = getattr(kargs, 'section_t_skin', 0.02*np.ones(self.n_stations))
        section_theta = getattr(kargs, 'section_theta', 10.*np.pi/180.*np.ones(self.n_stations))

        # Ensure all arguments are iterable 
        if not isinstance(section_chords, Iterable):
            section_chords = section_chords*np.ones(self.n_stations)
        if not isinstance(section_airfoils, Iterable):
            section_airfoils = [copy.deepcopy(section_airfoils) for _ in range(self.n_stations)]
        if not isinstance(section_t_skin, Iterable):
            section_t_skin = section_t_skin*np.ones(self.n_stations)
        if not isinstance(section_theta, Iterable):
            section_theta = section_theta*np.ones(self.n_stations)

        for prop in [section_chords, section_airfoils, section_t_skin, section_theta]:
            assert len(prop) == self.n_stations, (
                "Must supply either a singleton or iterable of same length when supplying"
                + " a section property"
            )
        
        # Create the station list
        self._blade_sections = [
            self.BladeSection(*args, **kargs) for args in zip(
                section_airfoils, section_chords, section_theta, section_t_skin,
                Nb*np.ones(self.n_stations), R*np.ones(self.n_stations)
            )
        ]

        # Create the profile functions
        self.bemt.profiles = dict()
        self.fem.profiles = dict()
        self.update_profiles()

        
    def update_sections(self, **kargs):
        """ Arbitrarily handling the section updates for the problem """
        for param in self.problem_params:
            if param in kargs.keys():
                val = kargs[param]
                if isinstance(val, Iterable):
                    assert len(val) == self.n_stations, (
                        "Iterable update must have same length as number of stations."
                    )
                else:
                    val = val*np.ones(self.n_stations)

                for bs, vi in zip(self._blade_sections, val):
                    setattr(bs, param, vi)

    def generate_interpolation_function(self, section_property, flag_dimensional=False):
        x = self.r_profile_stations
        if flag_dimensional:
            x *= self.R
        y = np.array([getattr(bs, section_property) for bs in self._blade_sections])
        sig = '()->(n)' if y.ndim > 1 else '()->()'
        func = np.vectorize(interp1d(x, y, axis=0), signature=sig)
        return func
        
    def update_profiles(self):
        for prop in self.aero_parameters:
            self.bemt.profiles.update({
                '{}_func'.format(prop) : self.generate_interpolation_function(prop)
            })
        self.bemt.update_blade_profile(**self.bemt.profiles)
        for prop in self.struct_parameters:
            self.fem.profiles.update({
                '{}_func'.format(prop) : self.generate_interpolation_function(prop, True)
            })

            
    def update_problem_from_state_vector(self, x):
        """
        
        Args:
            x (np.ndarray) : Optimization state vector of MDO problem
                           : vector of n_states_per_section*n_stations elements
        """

        X = x.reshape((self.n_stations, self.n_states_per_section))
        x_dict = dict()
        for j, param in enumerate(self.problem_params):
            x_dict.update({param : X[:,j]})

        self.update_sections(**x_dict)
        self.update_profiles()

        
    def generate_section_parameter_limits(
            self, state, sections=None, lower_vals=None, upper_vals=None
    ):
        """ Generates a list of inequality functions to ensure a state is bounded. """

        if isinstance(state, str):
            state = self.problem_params.index(state)

        if sections is None:
            sections = np.arange(self.n_stations)
            
        if not isinstance(sections, Iterable):
            sections = np.array([sections])

        n_sections = len(sections)
        
        if not isinstance(lower_vals, Iterable):
            if lower_vals is None:
                lower_vals = [None,]*n_sections
            else:
                lower_vals = lower_vals*np.ones(n_sections)
                
        if not isinstance(upper_vals, Iterable):
            if upper_vals is None:
                upper_vals = [None,]*n_sections
            else:
                upper_vals = upper_vals*np.ones(n_sections)

        # Iterate through sections
        inequality_list = list()
        for lv, uv, i in zip(
                lower_vals, upper_vals, (self.n_states_per_section*sections) + state
        ):
            j = copy.deepcopy(i)
            if lv is not None:

                fi_l = LowerBound(j, lv)
                inequality_list += [fi_l,]
                
            if uv is not None:
                
                fi_u = UpperBound(j, uv)
                inequality_list += [fi_u,]

        return inequality_list
                
            
    def generate_section_equalities(self, state, sections=None):
        """ Generates a list of inequalities for reducing parameter space """

        if isinstance(state, str):
            state = self.problem_params.index(state)

        if sections is None:
            sections = np.arange(self.n_stations)

        assert isinstance(sections, Iterable), "Must be giving multiple sections."
            
        # Anchor all inequalities on the first element
        i = state + self.n_states_per_section*sections[0]

        # Iterate through the rest of the elements 
        inequality_func_list = list()
        for nj in sections[1:]:
            j = state + self.n_states_per_section*nj

            def f_pos(x):
                return x[i] - x[j]

            def f_pos_grad(x):
                grad = np.zeros(x.shape)
                grad[i] = 1.; grad[j] = -1.
                return grad
            f_pos.gradient = f_pos_grad
            inequality_func_list += isaffineequality(f_pos)
            
        return inequality_func_list

    def generate_section_inequality_constraints(self, func):
        """ A functional inequality that must be upheald at eat cross-section """

        constraint_list = list()
        for i in range(self.n_stations):
            constraint_list.append(SectionConstraint(i, self.n_states_per_section, func))

        return constraint_list
    
    
if __name__=="__main__":

    # Design point, roughly based of a Robinson R-22
    blade_number = 2  # Two bladed rotor
    blade_radius = 3.84  # [m]
    thrust_requirement = 500.*9.81  # 500 [kg] vehicle in hover -> [N]
    power_limit = 90e3  # 90 [kW]
    rotor_omega = 2550.*(2.*np.pi)*(1./60.)  # [rpm] -> [rad/s]
    air_density = 1.225  # [kg/m^3]
    air_kinematic_viscosity = 1.729e-5

    # Material Properties for Aluminum 2024 T4
    # Ref: https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=ma2024t4
    material_E = 73.1e9  # 73.1 [GPa]
    material_G = 28e9  # 28 [GPa]
    material_density = 2.78*1e-3*1e6  # 2.78 [g/cc] into [kg/m^3]
    material_tensile_yield_stress = 324e6  # 324 [MPa]

    # Setup the MDO problem
    r_profile_stations = [0., 1.]
    n_elements_per_section = 50
    blade_constant_chord = 0.0075
    mdo_problem = RotorBladeAnalysis(
        blade_number, blade_radius, r_profile_stations, n_elements_per_section,
        rotor_omega, air_density,
        material_E, material_G,
    )
    x0 = np.array([0.075, 0.0075, 10./180.*np.pi, 0.002, 0.075, 0.0075, 10./180.*np.pi, 0.002])

    # Construct Affine Constraint List
    fi_affine = list()

    # Affine Equality Constraints Across Cross-Sections
    fi_affine += mdo_problem.generate_section_equalities('skin_thickness')

    # Affine Inequality Constraints
    fi_affine += mdo_problem.generate_section_parameter_limits(
        'chord', sections=None, lower_vals=0., upper_vals=0.2
    )  # Equality for constant solidity, needed for convexity of constraints
    fi_affine += mdo_problem.generate_section_parameter_limits(
        'theta', sections=None, lower_vals=0., upper_vals=20./180.*np.pi
    )  # Stall limits
    fi_affine += mdo_problem.generate_section_parameter_limits(
        'skin_thickness', sections=None, lower_vals=0.001
    )  # Manufactoring

    # Inequalities at each Cross-Section
    for f in [
            add_gradient(lambda x : x[3] - 0.5*x[1], lambda x : np.array([0., -0.5, 0., 1.])),
            add_gradient(lambda x : 0.08*x[0] - x[1], lambda x : np.array([0.08, -1., 0., 0.])),
            add_gradient(lambda x : x[1] - 0.2*x[0], lambda x : np.array([-0.2, 1., 0., 0.]))
    ]:
        # Inequalities from manufactoring
        fi_affine += mdo_problem.generate_section_inequality_constraints(isaffine(f))
    
    
    def f_thrust(x):
        mdo_problem.update_problem_from_state_vector(x)
        CT = mdo_problem.bemt.get_nondimensional_blade_performance()[0]
        return mdo_problem.bemt.dimensionalize_CT(CT, air_density, blade_number, rotor_omega)

    f_thrust = numerical_gradient(f_thrust, delta_x=1e-6)

    def fi_thrust(x):
        return thrust_requirement - f_thrust(x)

    fi_thrust.gradient = lambda x : -f_thrust.gradient(x)
    
    # Objective function
    @isaffine
    def obj_minimize_pitch(x):
        i = mdo_problem.problem_params.index('theta')
        return np.sum(x[i::mdo_problem.n_states_per_section])

    def obj_mp_grad(x):
        i = mdo_problem.problem_params.index('theta')
        grad = np.zeros(x.shape)
        grad[i::mdo_problem.n_states_per_section] = 1.
        return grad

    obj_minimize_pitch.gradient = obj_mp_grad

    # Learning Rates
    learning_rate = 1e-3
    momentum = 1e-1

    """
    # Test Optimization
    fig, ax = plt.subplots(2, 1, sharey=True, sharex=True)
    fig.set_size_inches((6.5, 4))
    ax[0].set_position([0.1, 0.57, 0.86, 0.37])
    ax[1].set_position([0.1, 0.11, 0.86, 0.37])
    
    opt_kargs = {
        'objective_function' : obj_minimize_pitch,
        'learning_rate' : learning_rate,
        'momentum' : momentum,
        'inequality_constraints' : fi_affine + [fi_thrust,]
    }

    for i, optclass in enumerate([GradientOptimizer, NesterovOptimizer]):

        optimizer = optclass(**opt_kargs)
        results = optimizer.optimize(
            x0,
            gradient_stop_criteria=-np.inf,
            min_step=-np.inf
        )
        
        xh = results['xh']
        iter_plot = np.arange(xh.shape[0])

        ax[i].plot(iter_plot, xh[:,2]*180./np.pi, label='Root')
        ax[i].plot(iter_plot, xh[:,6]*180./np.pi, label='Tip')

        ax[i].set_ylabel("Pitch Angle ($\\theta$) [deg]")
        ax[i].set_ylim(5, 15)

        ax[i].grid(True, which='major', axis='both')

    ax[0].set_title("Gradient Descent")
    ax[1].set_title("Nesterov Descent")
    ax[-1].set_xlabel("Iterations")
    ax[-1].set_xlim(left=0, right=iter_plot[-1])
    ax[0].legend(framealpha=1.)
    
    plt.show()
    """

    # MDO Problem
    @numerical_gradient
    def f_vonmises_constraint(x):
        mdo_problem.fem.apply_profiles()
        mdo_problem.update_problem_from_state_vector(x)
        F_vec = mdo_problem.bemt.get_dimensional_blade_element_forces_and_moments()
        F_vec = mdo_problem.fem.translate_aerodynamic_forces_and_moments_to_structural(F_vec)
        sigma2 = mdo_problem.fem.critical_vonMises_stress(F_vec.reshape((F_vec.size, 1)))

        return sigma2 - material_tensile_yield_stress**2

    
    # Objective function
    @isaffine
    def obj_minimize_weight(x):
        c = 0.
        i = mdo_problem.problem_params.index('thickness')
        c += np.sum(x[i::mdo_problem.n_states_per_section])
        i = mdo_problem.problem_params.index('skin_thickness')
        c += np.sum(x[i::mdo_problem.n_states_per_section] ** 2)
        return c

    def obj_weight_grad(x):
        grad = np.zeros(x.shape)
        i = mdo_problem.problem_params.index('thickness')
        grad[i::mdo_problem.n_states_per_section] = 1.
        i = mdo_problem.problem_params.index('skin_thickness')
        grad[i::mdo_problem.n_states_per_section] += 2.*x[i::mdo_problem.n_states_per_section]
        return grad

    obj_minimize_weight.gradient = obj_weight_grad

    x0 = np.array([0.2, 0.03, 10./180.*np.pi, 0.01, 0.2, 0.03, 10./180.*np.pi, 0.01])
    
    # Test Optimization
    fig, ax = plt.subplots(2, 1, sharey=True, sharex=True)
    fig.set_size_inches((6.5, 4))
    ax[0].set_position([0.1, 0.57, 0.86, 0.37])
    ax[1].set_position([0.1, 0.11, 0.86, 0.37])
    
    opt_kargs = {
        'objective_function' : obj_minimize_pitch,
        'learning_rate' : 1e-2, # learning_rate,
        'momentum' : momentum,
        'inequality_constraints' : fi_affine + [fi_thrust,] + [f_vonmises_constraint]
    }

    for i, optclass in enumerate([GradientOptimizer, NesterovOptimizer]):

        optimizer = optclass(**opt_kargs)
        results = optimizer.optimize(
            x0,
            gradient_stop_criteria=-np.inf,
            min_step=-np.inf
        )

        print(results)
        
        xh = results['xh']
        iter_plot = np.arange(xh.shape[0])

        ax[i].plot(iter_plot, xh[:,2]*180./np.pi, label='Root')
        ax[i].plot(iter_plot, xh[:,6]*180./np.pi, label='Tip')

        ax[i].set_ylabel("Pitch Angle ($\\theta$) [deg]")
        ax[i].set_ylim(5, 15)

        ax[i].grid(True, which='major', axis='both')

    ax[0].set_title("Gradient Descent")
    ax[1].set_title("Nesterov Descent")
    ax[-1].set_xlabel("Iterations")
    ax[-1].set_xlim(left=0, right=iter_plot[-1])
    ax[0].legend(framealpha=1.)
    
    plt.show()
