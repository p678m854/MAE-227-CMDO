# Convex Multidisciplinary Optimization

This repository contains code used for Mutlidisciplinary Optimization (MDO) of a rotorcraft blade using simple linear structures and Blade Element Momentum (BEMT). The code is broken down into a couple of sections, namely the 
1. aifoils used,
2. BEMT code
3. FEM code
4. general optimizer code
5. MDO Problem

Most of the files associated with this project can also be run as a script to generate intertesting results or validation plots/outputs. This code repository was developed for MAE 227 Convex optimization at the University of California, San Diego in the Spring 2023 quarter.

## Airfoils

This code only considers the airfoil cross sectional properties all code for this can be found in ```airfoils.py```. Currently only NACA 4-Series symmetric airfoils are considered but the class will calculate various properties such as

1. Aerodynamic properties by thickness
2. Solid airfoil structural properties by chord and thickness.
3. Hollow airfoil structural properties by chord and thickness.

## Blade Element Moment Theory

The code for BEMT is found ```bemt.py``` and closely follows Leishman's "Principles of Helicopter Aerodynamics". It deals exclusively in nondimensional thrust, power, and inflow for a rotor disk with zero advance ratio. Real effects such as tip losses have also been included in this code.

## Finite Element Model

The blade is modeled by finite elements using Euler-Bernoulli beam theory. The main FEM code is found in ```fem.py``` with a dditional codes associated with the development and testing. These additional codes are 

1. ```fem_constraints.py``` for the constraints.
2. ```fem_functions.py``` for objective functions
3. ```fem-optimizer.py``` to show the optimization of the structural discipline prolbem.
4. ```fem-test.py``` to validate the code.


## Optimizer

```optimizer.py``` contains an abstract base class optimizer for general optimization. The main steps to the optimizer in the optimization loop are 

1. Choosing a search direction
2. Ensuring the search direction is feasible
3. Ensuring a feasible step along the search direction
4. Update the estimated optimal point

By allowing the abstract base class to handle the main steps of optimization, derivative classes can simply change how the search direction is defined for rapid development testing of new optimization algorithms.

## MDO Problem

The main code for this project is ```mdo_problem.py``` which handles the multidisciplinary interactions between the objective functions and the constraints. It is quite the exhaustive code and the more practical parts are found after the ```___name__=="__main__"``` condtional check.