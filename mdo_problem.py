#!/usr/bin/python3

"""
Brief: Setup of the Multidisciplinary Design Optimization problem for the design
       of a helicopter rotor blade.
"""

from airfoils import NACA4Series
from bemt import BEMTAnalysis
from fem import FiniteElementModel
import numpy as np
from optimizer import affine, numerical_gradient, GradientOptimizer

def RotorBladeAnalysis:
    """ Class to handle rotor blade analysis """

    def __init__(
            self,
            Nb, R, r_profile_stations, n_elements_per_section,  # Blade parameters
            Omega,  # Aerodynamics parameter
            E, G  # Structural Parameter
            **kargs
    ):

        self.Nb = Nb
        self.R = R
        self.r_profile_stations
        self.n_elements_per_section
        self.rho = rho
        self.E = E
        self.G = G

        t_airfoil = getattr(kargs, 't_airfoil')
        self._airfoil_sections = [
            NACA4Series()
        ]
        
        
    
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
    
    # Affine Constraints
    
    
