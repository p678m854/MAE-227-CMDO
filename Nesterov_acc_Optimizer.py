# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 20:44:00 2023

@author: Devyansh
"""
#!/usr/bin/python3

"""
Brief: Handling of optimization
"""

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, Callable


def bisection_zero_search(x_l, x_r, f, max_iter=100, epsilon=1e-6):
    """ Zero searching on line interval
    
    Args:
        x_l (float) : left interval end point
        x_r (float) : right interval end point
        f (Callabe) : continuous function to search
        max_iter (int) : maximum number of iterations
np.linalg.norm(x_kp1 - x_k)        epsilon (float) : width tolerance of solution
    
    Returns:
        x_0 (float) : estimate of zero within 
    """
    
    fl = f(x_l)
    fr = f(x_r)

    assert np.sign(fl)*np.sign(fr) < 0, (
        "Function evalution must be opposite signs on interval endpoints."
    )

    n_iter = 0
    while (np.abs(x_r - x_l) > epsilon) and (n_iter < max_iter):
        n_iter += 1
        x_c = 0.5*(x_l + x_r)
        fc = f(x_c)

        if fc == 0.:
            return x_c
        elif np.sign(fc)*np.sign(fl) > 0:
            x_l = x_c
            f_l = fc
        else:
            x_r = x_c
            f_r = fc

    return x_c


def feasible_step_line_search(
        x_old, x_new,
        inequality_constraints,
        violation_tolerance,
        max_search_iter=100,
        search_step_size_tolerance=1e-6
):
    """ Makes searches along the line x(t) = t*x_old + (1-t)*x_new to find a feasible solution """

    t_feasible = 1.  # Unadulterated new point
    line_fun = lambda t : (1. - t)*x_old + t*x_new  # line function

    # Ensure feasibility over all the constraints
    for fi in inequality_constraints:
        # Check to see if step is feasible
        # print(fi(t_feasible))
        if fi(line_fun(t_feasible)) > violation_tolerance:
            t_feasible = bisection_zero_search(
                0., t_feasible,
                lambda tau : fi(line_fun(tau)),
                max_iter=max_search_iter,
                epsilon=search_step_size_tolerance
            )
        else:
            continue

    return line_fun(t_feasible)


def numerical_gradient(func, delta_x=1e-2):
    def central_difference_second_order(f, x, delta_x):
        fx = f(x)
        flag_different_steps = isinstance(delta_x, Iterable)
        dxi = delta_x
        gradient = np.zeros(x.shape, dtype=np.float64)  # Preassign gradient
        for i in range(x.size):
            if flag_different_steps:
                dxi = delta_x[i]

            xr = x.copy(); xr[i] += dxi
            xl = x.copy(); xl[i] -= dxi

            gradient[i] = (f(xr) - f(xl))/(2.*dxi)
            
            del xr, xl
        
        return gradient
    
    func.gradient = lambda x : central_difference_second_order(func, x, delta_x)
    return func


def isaffine(func, val=True):
    func.isaffine = val
    return func

@numerical_gradient
def foo(x):
    return np.linalg.norm(x)

class Optimizer(ABC):
    """ Abstract Optimizer Class

    Constructs the abstract class for various optimizaters. Inheriting classes are
    mainly responsible for determining the search direction while this class sets
    up the basics for optimization.
    """
    
    inequality_constraint_epsilon = 1e-6  # epsilon check of zero
    
    gradient_stop_criteria = 1e-4  # gradient magnitude stop criteria
    
    @abstractmethod
    def search_direction(self, x, *args, **kargs):
        """ Search direction is determined by optimizer """
        raise NotImplemented

    @property
    def objective_function(self):
        return self.__objective_function

    @objective_function.setter
    def objective_function(self, val):
        assert isinstance(val, Callable), "Objective function must be a callable object."
        self._is_valid_objective_function(val)
        self.__objective_function = val

    @abstractmethod
    def _is_valid_objective_function(self, f0):
        raise NotImplemented
    
    @property
    def inequality_constraints(self):
        return self.__inequality_constraints

    @inequality_constraints.setter
    def inequality_constraints(self, val):
        if isinstance(val, Callable):
            self.__inequality_constraints = [val,]
        elif isinstance(val, Iterable):
            for vi in val:
                assert isinstance(vi, Callable), "All inequality constraints must be functions"
                self._is_valid_constraint_function(vi)

            self.__inequality_constraints = list(val)

        else:
            raise ValueError(
                "Inequality constraints must be either a function or group of functions"
            )

    @abstractmethod
    def _is_valid_constraint_function(self, f0):
        raise NotImplemented
        
    def active_constraints(self, x):
        return [
            fi for fi in self.inequality_constraints if (
                fi(x) >= -self.inequality_constraint_epsilon
            )
        ]

    def violated_constraints(self, x):
        return [
            fi for fi in self.inequality_constraints if (
                fi(x) > self.inequality_constraint_epsilon
            )
        ]
    
    def feasible_direction(
            self, x : np.ndarray,
            search_direction : np.ndarray,
            epsilon_back_projection=1e-2
    ):
        """ Ensure search direction is feasible """
        
        feasible_direction = search_direction.copy()  # Assume search is initially feasible.

        # For every active constraint
        for fi in self.active_constraints(x):
            # Get the gradient of constraint
            grad_fi = fi.gradient(x)
            norm_grad_fi = np.linalg.norm(grad_fi)

            # Check if we need to project feasible direction
            direction_onto_gradient = np.dot(grad_fi, feasible_direction)
            if direction_onto_gradient > 0:
                # Do a projection into a safe direction
                feasible_direction -= grad_fi*direction_onto_gradient/(norm_grad_fi**2)
                if not getattr(fi, 'isaffine', False):
                    # For non-affine inequalities, we add some direction back into the domain
                    feasible_direction += (
                        -epsilon_back_projection*grad_fi/norm_grad_fi
                    )
            else:
                # Current direction is reducing the inequality function value
                continue
            
        return feasible_direction    

    @property
    def feasible_step_search(self):
        return feasible_step_line_search
    
    def ensure_feasible_update(self, x_old, x_new):
        """ Determines a feasible step from x_old to an approximate x_new """

        x_feasible = x_new.copy()
        violated_constraints = self.violated_constraints(x_feasible)
        if len(violated_constraints) > 0:
            x_feasible = self.feasible_step_search(
                x_old, x_feasible,
                violated_constraints, self.inequality_constraint_epsilon
            )

        return x_feasible

    
    def optimize(
            self, x0,
            gradient_stop_criteria=gradient_stop_criteria,
            max_iter=100,
            min_step=1e-6,
            verbose=1,
            **kargs
    ):

        # Assumes feasible initial set
        
        # Set up set counter and tracking states
        x_k = x0.copy()
        n_iter = 0
        n_variables = x0.size
        results = {
            'xh' : x_k.copy(),
            'd_desired' : np.empty((0, n_variables)),
            'd_feasible' : np.empty((0, n_variables)),
            'g_feasible_magnitude' : np.empty((0,1)),
            'n_active' : np.array([len(self.active_constraints(x0))])
        }

        # Helper function for fast results updates
        result_appender = lambda key, val : results.update({key : np.vstack((results[key], val))})
        
        # TODO: Add in prints here

        while n_iter < max_iter:

            # 1. Determine initial direction for search
            d_k = self.search_direction(
                x_k,
                # n_iter=n_iter,
                # d_prev=results['d_feasible']
            )

            result_appender('d_desired', d_k.copy())

            # 2. Change direction into a feasible one
            d_k = self.feasible_direction(x_k, d_k); result_appender('d_feasible', d_k.copy())
            # Note: d_k aready incorporates the magnitude of search

            # 3. Find feasible step in feasible direction
            try:
                x_kp1 = self.ensure_feasible_update(x_k, x_k + d_k)
            except AssertionError:
                results['sucess'] = False
                results['message'] = "Feasible direction could not be found."
                return results
            except Exception as e:
                results['success'] = False
                results['messeage'] = (
                    "Error occured trying to find a feasible search direction."
                    + "{!s}: {!s}".format(type(e), e)
                )
            
            # 3. Check convergence criteria
            g = self.objective_function.gradient(x_kp1)
            g_mag = np.linalg.norm(g)
            g_feas_mag = np.linalg.norm(self.feasible_direction(x_kp1, g))
            step_mag = np.linalg.norm(x_kp1 - x_k)

            result_appender('g_feasible_magnitude', g_feas_mag)
            
            # 3.a. Gradient magnitude
            if g_feas_mag < gradient_stop_criteria:
                result_appender('xh', x_kp1.copy())
                results['success'] = True
                results['message'] = (
                    "Terminated as feasible gradient magnitude fell below threshold."
                )

                return results
            
            # 3.b. Minimum step criteria
            if step_mag < min_step:
                result_appender('xh', x_kp1.copy())
                results['success'] = True
                results['message'] = "Terminated as step magnitude fell below threshold."

                return results
                
            # Update variables
            x_k = x_kp1.copy(); result_appender('xh', x_kp1.copy())
            result_appender('n_active', len(self.active_constraints(x_k)))
            n_iter += 1

        results["success"] = True
        results['message'] = "Terminated as maximum iterations reached."
        return results

    
class GradientOptimizer(Optimizer):
    """ Gradient Descent Algorithm """

    def __init__(
            self,
            objective_function,
            learning_rate,
            inequality_constraints=list()
    ):

        self.learning_rate = learning_rate
        self.objective_function = objective_function
        self.inequality_constraints = inequality_constraints

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, val):
        self.__learning_rate = val
        
    def search_direction(self, x, *args, **kargs):
        n_iter = kargs.get("n_iter", 'not passed')  # Not necessary here but an example of kargs
        if np.isscalar(self.learning_rate):
            return -self.learning_rate*self.objective_function.gradient(x)
        else:
            return -self.learning_rate @ self.objective_function.gradient(x)

    @classmethod
    def _function_has_gradient(cls, f):
        return hasattr(f, 'gradient')

    def _is_valid_objective_function(self, f0):
        if self._function_has_gradient(f0):
            return True
        else:
            raise ValueError("Objective function must have a defined gradient.")

    def _is_valid_constraint_function(self, fi):
        if self._function_has_gradient(fi):
            return True
        else:
            raise ValueError("Constraint function must have a defined gradient.")


if __name__=="__main__":
    
    # Construct the constraints
    circle_center = np.array([0., 0.])
    circle_radius = 5.

    fi_circle = lambda x : np.linalg.norm(x - circle_center)**2 - circle_radius**2
    fi_circle.gradient = lambda x : 2.*np.eye(2) @ (x - circle_center)
    
    plane_norm_dir = np.array([1., 0.])
    plane_norm_dir = plane_norm_dir/(np.linalg.norm(plane_norm_dir)**2)
    plane_perp_dir = np.array([1., 1.]) - plane_norm_dir*np.sum(plane_norm_dir)
    plane_dist = 2.

    fi_plane = lambda x : np.dot(plane_norm_dir, x) - plane_dist
    fi_plane.gradient = lambda x : plane_norm_dir
    fi_plane = isaffine(fi_plane)

    # Objective function
    @numerical_gradient
    def f0(x):
        return -x[0] + x[1]

  
class NesterovOptimizer(Optimizer):
    """ Nesterov's Accelerated Gradient Descent Algorithm """

    def __init__(
            self,
            objective_function,
            learning_rate,
            momentum,
            inequality_constraints=list()
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.objective_function = objective_function
        self.inequality_constraints = inequality_constraints
        self.previous_direction = None

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, val):
        self.__learning_rate = val

    @property
    def momentum(self):
        return self.__momentum

    @momentum.setter
    def momentum(self, val):
        self.__momentum = val

    def search_direction(self, x, *args, **kwargs):
        n_iter = kwargs.get("n_iter", 'not passed')  # Not necessary here but an example of kwargs
        if self.previous_direction is None:
            self.previous_direction = -self.learning_rate * self.objective_function.gradient(x)
            return self.previous_direction
        else:
            current_direction = -self.learning_rate * self.objective_function.gradient(x) + \
                                self.momentum * self.previous_direction
            self.previous_direction = current_direction
            return current_direction

    @classmethod
    def _function_has_gradient(cls, f):
        return hasattr(f, 'gradient')

    def _is_valid_objective_function(self, f0):
        if self._function_has_gradient(f0):
            return True
        else:
            raise ValueError("Objective function must have a defined gradient.")

    def _is_valid_constraint_function(self, fi):
        if self._function_has_gradient(fi):
            return True
        else:
            raise ValueError("Constraint function must have a defined gradient.")

# Get the optimizer for the problem
eta = 0.05
momentum = 0.9
opt_nesterov = NesterovOptimizer(f0, eta, momentum, [fi_circle, fi_plane])
# Figure for plotting results
fig, ax = plt.subplots(1,1)

# Iterate through some starting conditions
flag_trajectory_label = True
for x0 in np.array(
        [
            [-4., 0.],
            [-3, -3],
            [0, 4]
        ]
):

    ax.scatter(
        x0[:1], x0[1:], s=14**2., color='c', marker='.',
        label=('Starting Points' if flag_trajectory_label else '_'),
        zorder=3
    )
    
    result = opt_nesterov.optimize(x0, max_iter=200)
    xh = result['xh']
    
    ax.plot(
        xh[:,0], xh[:,1], color='b', linewidth=2.,
        label=("Nesterov Trajectories" if flag_trajectory_label else "_")
    )

    ax.scatter(
        xh[-1,:1], xh[-1,1:], s=12**2., color='c', marker='h',
        label=('Stopping Points' if flag_trajectory_label else '_'),
        zorder=3
    )
    
    flag_trajectory_label = False

# Add in constraints
theta_p = np.linspace(0, 2.*np.pi)
ax.plot(
    circle_radius*np.cos(theta_p) + circle_center[0],
    circle_radius*np.sin(theta_p) + circle_center[1],
    color='k', label="Circle Constraint", linewidth=2.
)

length_p = 5.
ax.plot(
    length_p*np.array([-1, 1])*plane_perp_dir[0] + plane_dist*plane_norm_dir[0],
    length_p*np.array([-1, 1])*plane_perp_dir[1] + plane_dist*plane_norm_dir[1],
    label="Half Space Constraint", color='r', linewidth=2.
)

ax.legend(framealpha=1.)
plt.show()
    

# Add in constraints
theta_p = np.linspace(0, 2.*np.pi)
ax.plot(
    circle_radius*np.cos(theta_p) + circle_center[0],
    circle_radius*np.sin(theta_p) + circle_center[1],
    color='k', label="Circle Constraint", linewidth=2.
)

length_p = 5.
ax.plot(
    length_p*np.array([-1, 1])*plane_perp_dir[0] + plane_dist*plane_norm_dir[0],
    length_p*np.array([-1, 1])*plane_perp_dir[1] + plane_dist*plane_norm_dir[1],
    label="Half Space Constraint", color='r', linewidth=2.
)



    