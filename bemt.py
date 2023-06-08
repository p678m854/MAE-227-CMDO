#!/usr/bin/python3

"""
Brief: Blade Element Momentum Theory Code
"""


import matplotlib.pyplot as plt
import numpy as np


def prandtl_f_function(
    r_t, r_b, phi_vec, Nb
):
    f = Nb/2.*np.divide(r_t, (r_b*phi_vec), out=np.zeros(r_t.shape), where=(r_b > 0))
    f = np.where(r_b > 0, f, np.inf)
    return f
    

def prandtl_correction_factor(r_vec, phi_vec, Nb, r_hub=0., flag_root_effects=False):
    r_t = 1. - r_vec
    r_b = r_vec - r_hub
    f_tip = prandtl_f_function(r_t, r_b, phi_vec, Nb)
    

    F = (2./np.pi)*np.arccos(np.exp(-f_tip))
    

    if flag_root_effects:
        f_root = prandtl_f_function(r_b, r_t, phi_vec, Nb)
        F_root = (2./np.pi)*np.arccos(np.exp(-f_root))

        F *= F_root

    return F


def nondimensional_inflow(r_vec, theta_vec, C_l_alpha_vec, sigma_vec, F_vec=1., lambda_c=0.):
    """
    Eq. 3.131
    """
    a = sigma_vec*C_l_alpha_vec/(16.*F_vec)  - lambda_c/2.
    b = (sigma_vec*C_l_alpha_vec)/(8.*F_vec)*theta_vec*r_vec
    return np.power(np.power(a, 2.) + b, 0.5) - a


def incremental_nondimensional_thrust_coefficient(
    r_vec, theta_vec, C_l_alpha_vec, sigma_vec, lambda_vec
):
    """
    Eq. (3.72)
    """
    
    dCT_dr = sigma_vec/2.*C_l_alpha_vec*(
        theta_vec*r_vec*r_vec - lambda_vec*r_vec
    )

    return dCT_dr


def incremental_induced_nondimensional_power_coefficient(lambda_vec, dCT_dr_vec):
    """
    Eq. (3.74)
    """

    return lambda_vec*dCT_dr_vec


def incremental_profile_nondimensional_power_coefficient(
    r_vec, C_d_0_vec, d_1_vec, d_2_vec, lambda_vec, theta_vec, sigma_vec
):
    """
    Eq. (3.117)
    """
    
    return 0.5*sigma_vec*(
        C_d_0_vec*r_vec*r_vec*r_vec  # Parasitic drag
        + d_1_vec*(theta_vec*r_vec - lambda_vec)*r_vec*r_vec  # linear part of drag profile
        + d_2_vec*np.power(theta_vec*r_vec - lambda_vec, 2.)*r_vec  # quadratic part of profile
    )
        
# lambda = phi * r

def incremental_2D_lift_coefficient(r_vec, theta_vec, lambda_vec, C_l_alpha_vec):
    return C_l_alpha_vec*(theta_vec - lambda_vec/r_vec)

def incremental_blade_quarter_chord_pitching_coefficient(
        r_vec, C_m_c4_alpha_vec, lambda_vec, theta_vec, sigma_vec, Nb
):
    """
        d M_x_{c/4} = (1/2*rho*V**2)*(c**2)*(c_{m_{c/4_{\alpha}}} \alpha)
        nondimensionalize by (rho*A*V_tip**2*R)
    """
    alpha_r_vec = theta_vec*r_vec - lambda_vec
    return 0.5*np.pi/(Nb*Nb)*sigma_vec*sigma_vec*(C_m_c4_alpha_vec*alpha_r_vec)*r_vec*r_vec

def calculate_blade_thrust(
    r_vec, theta_vec, C_l_alpha_vec, sigma_vec, delta_r_vec,
    lambda_c=0., flag_tip_effects=False, Nb=2, r_hub=0.,
    **kargs
):

    # Set up loop flag
    flag_no_convergence = True
    
    # assume no tip losses
    F_vec = np.ones(r_vec.shape)
    F_vec_prev = F_vec.copy()

    
    while flag_no_convergence:
        
        # Determine the inflow
        lambda_vec = nondimensional_inflow(
            r_vec, theta_vec, C_l_alpha_vec, sigma_vec, F_vec=F_vec, lambda_c=lambda_c
        )

        # Angle of inflow (using small angles)
        phi_vec = np.divide(
            lambda_vec, r_vec,
            out=np.zeros(lambda_vec.shape), where=(r_vec > 0)
        )# lambda_vec/r_vec
        phi_vec = np.where(r_vec > 0, phi_vec, np.inf)

        # Find thrust from inflow
        dCT_dr = incremental_nondimensional_thrust_coefficient(
            r_vec, theta_vec, C_l_alpha_vec, sigma_vec, lambda_vec
        )

        # Recalculate Prandtl Factor
        if flag_tip_effects:
            F_vec = prandtl_correction_factor(r_vec, phi_vec, Nb, r_hub)

        # Check convergence
        if np.sum(np.abs(F_vec - F_vec_prev)) < 1e-4:
            flag_no_convergence = False
        else:
            F_vec_prev = F_vec.copy()

    # Integrate the thrust
    C_T = np.sum(dCT_dr*delta_r_vec)
            
    return C_T, dCT_dr, lambda_vec


def calculate_blade_power(
    r_vec, theta_vec, sigma_vec, delta_r_vec, dCT_dr_vec, lambda_vec,
    C_d_0_vec, d_1_vec, d_2_vec, **kargs
):
    # Find power from inflow and thrust
    dCPi_dr_vec = incremental_induced_nondimensional_power_coefficient(lambda_vec, dCT_dr_vec)
    dCP0_dr_vec = incremental_profile_nondimensional_power_coefficient(
        r_vec, C_d_0_vec, d_1_vec, d_2_vec, lambda_vec, theta_vec, sigma_vec
    )
    dCP_dr_vec = dCPi_dr_vec + dCP0_dr_vec

    # Integrate power
    C_P = np.sum(dCP_dr_vec*delta_r_vec)

    return C_P, dCP_dr_vec

def calculate_blade_pitching_moment(
        r_vec, theta_vec, sigma_vec, delta_r_vec, lambda_vec, C_m_c4_alpha_vec, Nb, **kargs
):
    dCmc4_dr = incremental_blade_quarter_chord_pitching_coefficient(
        r_vec, C_m_c4_alpha_vec, lambda_vec, theta_vec, sigma_vec, Nb
    )
    return np.sum(dCmc4_dr*delta_r_vec), dCmc4_dr
    

def trim_blade_pitch(
    C_T_required, r_vec, theta_rel_vec, C_l_alpha_vec, sigma_vec, delta_r_vec,
    lambda_c=0., flag_tip_effects=False, Nb=2, r_hub=0., tol=1e-7, max_iter=1000, verbose=0
):

    # Some representative values for quick numerics
    sigma_rep = np.median(sigma_vec)
    C_l_alpha_rep = np.median(C_l_alpha_vec)

    # Get a twist for an initial guess
    theta_tw = theta_rel_vec[-1] - theta_rel_vec[0]  # Assumes higher pitches towards the front
    
    theta_0 = (
        6.*C_T_required/(sigma_rep*C_l_alpha_rep)
        -3.*theta_tw/4. + 3.*np.sqrt(2*C_T_required)/4.
    )  # Initial guess based on a linear twisted blade

    # Print header if appropriate
    if verbose > 0:
        print("{: <12}{: >20}{: >16}".format("Iteration", "theta(r_hub) [rad]", "C_T"))
        print("{:=^69}".format(''))
    
    # Iterate till convergence
    n_iter = 0
    while n_iter < max_iter:
        # Calculate current thrust coefficient
        C_T, _, _ = calculate_blade_thrust(
            r_vec, theta_rel_vec + theta_0, C_l_alpha_vec, sigma_vec, delta_r_vec,
            lambda_c=lambda_c, flag_tip_effects=flag_tip_effects, Nb=Nb, r_hub=r_hub
        )

        if verbose > 0:
            print("{: >10}{: ^12}{:>01.7}{: ^8}{:>01.7f}".format(n_iter, '', theta_0, '', C_T))
            
        if np.abs(C_T - C_T_required) < tol:
            break
        else:
            theta_0 += (
                (6*(C_T_required - C_T))/(sigma_rep*C_l_alpha_rep)
                + (3*np.sqrt(2.)/4.)*(np.sqrt(C_T_required) - np.sqrt(C_T))
            )  # Eq. (3.77) for a linear twist blade

            n_iter += 1

    return theta_0


class BEMTAnalysis:
    """ Class to handle a BEMT analysis of a rotor blade for the MDO 

    Only set up to handle a linear tapered blade
    """

    update_profile_attributes = [
        'r_hub', 'n_elements'
    ]

    @classmethod
    def dimensionalize_CT(cls, CT, rho, R, Omega):
        return CT*rho*(np.pi*R*R)*(R*R)*(Omega*Omega)

    @classmethod
    def dimensionalize_CP(cls, CP, rho, R, Omega):
        return CP*rho*(np.pi*R*R)*(R*R*R)*(Omega*Omega*Omega)

    @classmethod
    def dimensionalize_CQ(cls, CQ, rho, R, Omega):
        return CQ*rho*(np.pi*R*R)*(R*R*R)*(Omega*Omega)

    @classmethod
    def dimensionalize_lambda(cls, lambda_vec, R, Omega):
        return lambda_vec*R*Omega

    @classmethod
    def dimensionalize_pitching_moment(cls, C_m_c4, rho, R, Omega):
        return C_m_c4*rho*(np.pi*R*R)*(Omega*Omega)*(R*R*R)
    
    def __init__(self, n_elements, Nb=2, r_hub=0., **kargs):

        self.r_hub = r_hub
        self.Nb = Nb
        self.n_elements = n_elements
        self.delta_r = (1. - r_hub)/n_elements
        self.func_kargs = kargs

    @property
    def r_vec(self):
        return self.__r_vec

    @r_vec.setter
    def r_vec(self, val):
        raise Exception("Unable to assign a radius vector, only for internals.")

    @property
    def delta_r(self):
        return self.__delta_r

    @delta_r.setter
    def delta_r(self, val):
        self.__delta_r = val
    
    @property
    def r_hub(self):
        return self.__r_hub

    @r_hub.setter
    def r_hub(self, val):
        assert 0 <= val < 1, "Hub radius must be in [0, 1)."
        self.__r_hub = val
        if hasattr(self, "n_elements"):
            self.__delta_r = (1. - self.r_hub)/self.n_elements
            self.__r_vec = np.arange(self.n_elements)*self.delta_r + 0.5*self.delta_r
            self.delta_r_vec = np.ones(self.n_elements)*self.delta_r
        if all([hasattr(val, attr) for attr in self.update_profile_attributes]):
            self.update_blade_profile(
                getattr(self, "sigma_func", None),
                getattr(self, "theta_func", None),
                getattr(self, "C_l_alpha_func", None),
                getattr(self, "C_d_0_func", None),
                getattr(self, "d_1_func", None),
                getattr(self, "d_2_func", None)
            )

    @property
    def n_elements(self):
        return self.__n_elements

    @n_elements.setter
    def n_elements(self, val):
        self.__n_elements = val

    @n_elements.setter
    def n_elements(self, val):
        assert val > 0, "Must have positive number of elements"
        self.__n_elements = val
        if hasattr(self, "r_hub"):
            self.__delta_r = (1. - self.r_hub)/self.n_elements
            self.__r_vec = np.arange(self.n_elements)*self.delta_r + 0.5*self.delta_r
            self.delta_r_vec = np.ones(self.n_elements)*self.delta_r
        if all([hasattr(val, attr) for attr in self.update_profile_attributes]):
            self.update_blade_profile(
                getattr(self, "sigma_func", None),
                getattr(self, "theta_func", None),
                getattr(self, "C_l_alpha_func", None),
                getattr(self, "C_d_0_func", None),
                getattr(self, "d_1_func", None),
                getattr(self, "d_2_func", None)
            )

    def update_blade_profile(
            self,
            **kargs
    ):
        """ Update the blade design 

        All inputs are key word arguments to functions that that are single parameter 
        functions of r.

        Args:
            sigma_func (Callable) : Blade solidity profile
            C_l_alpha_func (Callable) : Blade lift slope curve
            C_d_0_func (Callable) : Parasitic drag coefficient or drag at zero angle of attack.
            d_1_func (Callable) : Linear drag coefficient parameter function.
            d_2_func (Callable) : Quadratic drag coefficient parameter function.
            C_m_c4_alpha_func (callable) : Quarter chord pitching moment.
        """

        properties = [
            "sigma", 'theta',  # Designer blade decision variables
            'C_l_alpha',  # Airfoil lift parameters
            'C_d_0', 'd_1', 'd_2',  # Airfoil drag Parameters
            'C_m_c4_alpha'  # Airfoil quarter chord pitching moment
        ]
        
        for prop in properties:
        
            func_label = '{}_func'.format(prop)
            vec_label = '{}_vec'.format(prop)

            if func_label in kargs.keys():
                setattr(
                    self, func_label, np.vectorize(kargs[func_label], signature='()->()')
                )
                setattr(
                    self, vec_label, getattr(self, func_label)(self.r_vec)
                )

    def get_nondimensional_blade_performance(self):
        """ Gets the relavent nondimensional blade performance """

        C_T, dCT_dr, lambda_vec = calculate_blade_thrust(
            self.r_vec, self.theta_vec, self.C_l_alpha_vec, self.sigma_vec, self.delta_r_vec,
            **self.func_kargs
        )

        C_P, dCP_dr = calculate_blade_power(
            self.r_vec, self.theta_vec, self.sigma_vec, self.delta_r_vec,
            dCT_dr, lambda_vec,
            self.C_d_0_vec, self.d_1_vec, self.d_2_vec,
            **self.func_kargs
        )

        C_m_c4, dCmc4_dr = calculate_blade_pitching_moment(
            self.r_vec, self.theta_vec, self.sigma_vec, self.delta_r_vec,
            lambda_vec,
            self.C_m_c4_alpha_vec, self.Nb,
            **self.func_kargs
        )

        return C_T, C_P, dCT_dr, dCP_dr, dCmc4_dr, lambda_vec


    def get_dimensional_blade_element_forces_and_moments(self):
        """ Finds the forces and moments acting on element in rotor frame. """

        Omega = self.func_kargs['Omega']
        R = self.func_kargs['R']
        rho = self.func_kargs['rho']
        dCT_dr, dCP_dr, dCmc4_dr = self.get_nondimensional_blade_performance()[2:-1]
        dT_vec = self.dimensionalize_CT(dCT_dr*self.delta_r_vec, rho, R, Omega)/self.Nb
        dFD_vec = self.dimensionalize_CQ(dCP_dr*self.delta_r_vec, rho, R, Omega)/(
            self.Nb*R*self.r_vec
        )
        dCM_vec = self.dimensionalize_pitching_moment(dCmc4_dr*self.r_vec, rho, R, Omega)

        f = np.stack(
            (
                -dFD_vec, dT_vec, dCM_vec, np.zeros(self.n_elements), np.zeros(self.n_elements)
            ),
            axis=1
        )  # Beam element [F_y, F_z, M_x, M_y, M_z]

        return f

        
        
        
    
if __name__=="__main__":

    # Defaults for plotting
    C_T_desired = 0.008
    C_l_alpha = 5.73  # [~/rad]
    sigma = 0.1
    C_d_0 = 0.011
    
    # Recreate Figure 3.17
    fig, ax = plt.subplots(1,1)
    
    r_vec = np.linspace(0., 1., 51)  # np.linspace(0.7, 1.)
    r_vec = 0.5*(r_vec[1:] + r_vec[:-1])
    delta_r = r_vec[1] - r_vec[0]
    for Nb in np.array([2, 4]):
        for phi in np.array([0.2, 0.05]):

            F_vec = prandtl_correction_factor(r_vec, phi, Nb)

            ax.plot(r_vec, F_vec)

    ax.set_xlim(0.7, 1.)
    ax.set_ylim(0, 1)

    ax.set_title("Figure 3.17")
    ax.set_xlabel("Nondimensional radial position ($r$)")
    ax.set_ylabel("Prandtl's tip loss factor ($F$)")


    # Recreate Figure 3.5
    theta_0 = trim_blade_pitch(
        C_T_desired, r_vec, np.zeros(r_vec.shape), C_l_alpha, sigma, delta_r
    )
    lambda_vec = nondimensional_inflow(r_vec, theta_0, C_l_alpha, sigma)
    fig, ax = plt.subplots(1,1)

    ax.plot(r_vec, lambda_vec, label="No blade twist")
    ax.plot(
        r_vec, np.ones(r_vec.shape)*np.sqrt(C_T_desired/2),   # Eq. (3.66)
        label="Ideal blade twist", linestyle='--'
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.1)

    ax.set_title("Figure 3.5")
    ax.set_xlabel("Nondimensional radial posiiton, ($r$)")
    ax.set_ylabel("Local induced inflow ratio, ($\\lambda_i$)")
    ax.legend()

    # Recreate Figure 3.7 and Figure 3.8
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title("Figure 3.7")

    _, ax_cl = plt.subplots(1,1)
    ax_cl.set_title("Figure 3.8")

    # Plot Ideals
    lambda_ideal = np.sqrt(C_T_desired/2.)
    theta_tip_ideal = (4.*C_T_desired)/(sigma*C_l_alpha) + lambda_ideal
    ax[0].plot(r_vec, np.ones(r_vec.shape)*lambda_ideal, label="Ideal blade twist")
    ax[1].plot(
        r_vec, sigma/4.*C_l_alpha*(theta_tip_ideal - lambda_ideal)*r_vec,
        label="Ideal blade twist"
    )
    ax_cl.plot(r_vec, 4*C_T_desired/sigma*np.power(r_vec, -1), label="Ideal blade twist")
    
    # Iterate over blade twists
    for theta_twist in np.array([0., -10., -20.]):
        # Get new relative blade twist
        theta_vec = np.linspace(0., theta_twist/180.*np.pi, r_vec.size+1)
        theta_vec = 0.5*(theta_vec[1:] + theta_vec[:-1])

        # Trim blade
        theta_0 = trim_blade_pitch(
            C_T_desired, r_vec, theta_vec, C_l_alpha, sigma, delta_r
        )
        theta_vec += theta_0

        # Get performance
        _, dCT_dr, lambda_vec = calculate_blade_thrust(
            r_vec, theta_vec, C_l_alpha, sigma, delta_r,   
        )

        C_l_vec = incremental_2D_lift_coefficient(r_vec, theta_vec, lambda_vec, C_l_alpha)

        # Plot results
        ax[0].plot(
            r_vec, lambda_vec, label="${:>-2d}^\\circ$ linear twist".format(int(theta_twist))
        )
        ax[1].plot(r_vec, dCT_dr, label="${:>-2d}^\\circ$ linear twist".format(int(theta_twist)))
        ax_cl.plot(r_vec, C_l_vec, label="${:>-2d}^\\circ$ linear twist".format(int(theta_twist)))
        
    for i in range(len(ax)):
        ax[i].set_xlim(0, 1)
        ax[i].legend()
        ax[i].grid()

    ax[1].set_xlabel("Nondimensional radial position, ($r$)")
    ax[0].set_ylabel("Local induced inflow ratio, ($\\lambda_i$)")
    ax[1].set_ylabel("Thrust gradient, ($dC_T/dr$)")

    ax[0].set_ylim(0, 0.1)
    ax[1].set_ylim(0, 0.03)

    ax_cl.set_xlabel("Nondimensional radial position, ($r$)")
    ax_cl.set_ylabel("Local lift coefficient, ($C_l$)")

    ax_cl.set_xlim(0, 1.0)
    ax_cl.set_ylim(0, 2.0)

    ax_cl.legend()
    ax_cl.grid()
    
    # Recreate Figure 3.2 and 3.3
    fig, ax = plt.subplots(1,1)
    _, ax_cp = plt.subplots(1,1)

    delta_r = r_vec[1] - r_vec[0]
    theta_vec = np.linspace(0, 15./180.*np.pi)
    C_T_vec = np.zeros(theta_vec.shape)
    C_P_vec = np.zeros(theta_vec.shape)
    for sigma in np.array([0.042, 0.064, 0.085, 0.106]):

        for i, theta in enumerate(theta_vec):
            C_T, dCT_dr, lambda_vec = calculate_blade_thrust(
                r_vec, theta, C_l_alpha, sigma, delta_r
            )
            C_P, _ = calculate_blade_power(
                r_vec, theta, sigma, delta_r, dCT_dr, lambda_vec, C_d_0, 0., 0.
            )

            C_T_vec[i] = C_T
            C_P_vec[i] = C_P

        ax.plot(theta_vec*180./np.pi, C_T_vec, label="$\\sigma = {:1.3}$".format(sigma))
        ax_cp.plot(theta_vec*180./np.pi, C_P_vec, label="$\\sigma = {:1.3}$".format(sigma))

    ax.set_title("Figure 3.2")
    ax.set_xlabel("Blade-pitch angle, ($\\theta_0$) [deg]")
    ax.set_ylabel("Thrust coefficient, ($C_T$)")

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 0.015)
    
    ax.legend()
    ax.grid()

    ax_cp.set_title("Figure 3.3")
    ax_cp.set_xlabel("Blade-pitch angle, ($\\theta_0$) [deg]")
    ax_cp.set_ylabel("Power coefficient, ($C_P$)")

    ax_cp.set_xlim(0, 15)
    ax_cp.set_ylim(0, 0.0015)
    
    ax_cp.legend()
    ax_cp.grid()

    # Recreate Figure 3.12
    def generate_linear_rotor_solidity(sigma_e, taper_ratio, r_hub=0.):
        """
        Args:
            sigma_e : solidity at 80% span
            taper_ratio : tip chord over root chord
        """

        sigma_0 = sigma_e/(0.2 + 0.8/taper_ratio)
        a = sigma_0*(1./taper_ratio - 1)/(1. - r_hub)

        def linear_profile(r):
            if (r < r_hub) or (r > 1):
                return 0
            else:
                return sigma_0 + a*(r - r_hub)

        return linear_profile

    """
    fig, ax = plt.subplots(1,1)
    ax.set_title("Figure 3.12")

    theta_tw = -10./180.*np.pi
    sigma = 0.1
    C_T_desired = 0.008
    for tr in [1, 2, 3]:
        profile_fun = generate_linear_rotor_solidity(sigma, tr)
        sigma_vec = np.apply_along_axis(
            profile_fun,
            axis=1,
            arr=r_vec.reshape(r_vec.shape + (1,))
        )[:,0]
        theta_vec = np.linspace(0, theta_tw, r_vec.size + 1)
        theta_vec = 0.5*(theta_vec[1:] + theta_vec[:-1])

        theta_0 = trim_blade_pitch(
            C_T_desired, r_vec, theta_vec, C_l_alpha, sigma_vec, delta_r,
            flag_tip_effects=True, verbose=1
        )

        theta_vec += theta_0

        _, _, lambda_vec = calculate_blade_thrust(
            r_vec, theta_vec, C_l_alpha, sigma, delta_r
        )

        C_l_vec = incremental_2D_lift_coefficient(
            r_vec, theta_vec, lambda_vec, C_l_alpha
        )

        ax.plot(r_vec, C_l_vec, label="{}:1 linear taper".format(tr))

    ax.set_xlabel("Nondimensional radial position, ($r$)")
    ax.set_ylabel("Local lift coefficient, ($C_l$)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)
    ax.legend()
    ax.grid()
    """
    
    plt.show()
