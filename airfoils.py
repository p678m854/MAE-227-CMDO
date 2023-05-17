#!/usr/bin/python3

"""
"""

import numpy as np


class NACA4Series:

    # y_t(x) = 5*t*(a\sqrt{x} + b x + c x^2 + d x^3 + e x^4
    # y_t profile coefficients
    a = 00.2969  # sqrt(x)
    b = -0.1260  # x
    c = -0.3516  # x^2
    d = 00.2843  # x^3
    e = -0.1015  # x^4

    @classmethod
    def y_u_unit_chord(cls, x, t):
        return 5.*t*(cls.a*np.sqrt(x) + cls.b*x + cls.c*x*x + cls.d*x*x*x + cls.e*x*x*x*x)

    @classmethod
    def y_l_unit_chord(cls, x, t):
        return -5.*t*(cls.a*np.sqrt(x) + cls.b*x + cls.c*x*x + cls.d*x*x*x + cls.e*x*x*x*x)

    @classmethod
    def curvature_unit_chord(cls, x, t):
        return 5.*t*(
            cls.a*(-0.25)*np.power(x, -3./2.) + 2.*cls.c + 6.*cls.d*x + 12.*cls.e*x*x
        )
    
    def __init__(self, t):
        """
        Args:
            t (float) : thickness of airfoil (xx/100. in NACA YZXX)
        """

        self._t = t

        # Short hand the class variables
        a, b, c, d, e = NACA4Series.a, NACA4Series.b, NACA4Series.c, NACA4Series.d, NACA4Series.e
        
        """ Gather the unit chord properties """
        # Cross-sectional area
        self._unit_chord_area = 10.*t*(
            a*2./3. + b/2. + c/3. + d/4. + e/5.
        )

        # X position of center of area from LE
        self._unit_chord_x_area = (
            10.*t*(2.*a/5. + b/3. + c/4. + d/5. + e/6.)
        )/self._unit_chord_area

        # Bending moment of inertia about y-axis
        self._unit_chord_Iyy = (
            10.*t*(2.*a/7. + b/4. + c/5. + d/6. + e/7.)  # Ixx around tip
             - self._unit_chord_area*(self._unit_chord_x_area**2)  # Correct for offset
        )

        # Bending moment of inertia about x-axis
        self._unit_chord_Ixx = 2./3.*((5*t)**3)*(
            # one factor
            a*a*a*2./5. + b*b*b/4. + c*c*c/7. + d*d*d/10. + e*e*e/13.
            # two factors
            + 3.*(a*a*b/3. + a*a*c/4. + a*a*d/5. + a*a*e/6.)
            + 3.*(a*b*b*2./7. + b*b*c/5. + b*b*d/6. + b*b*d/7.)
            + 3.*(a*c*c*2./11. + b*c*c/6. + c*c*d/8. + c*c*e/9.)
            + 3.*(a*d*d*2./15. + b*d*d/8. + c*d*d/9. + d*d*e/11.)
            + 3.*(a*e*e*2./19. + b*e*e/10. + c*e*e/11. + d*e*e/12.)
            # three factors
            + 6.*(a*b*c*2./9. + a*b*d*2./11. + a*b*e*2./13.)
            + 6.*(a*c*d*2./13. + a*c*e*2./15. + a*d*e*2./17.)
            + 6.*(b*c*d/7. + b*c*e/8. + b*d*e/9. + c*d*e/10.)
        )
        self._unit_chord_J = self._unit_chord_Ixx + self._unit_chord_Iyy

    # Assumption: cutouts have the same profile
        
    def A(self, c):
        return c*c*self._unit_chord_area

    def x_center_area(self, c):
        return c*self._unit_chord_x_area
    
    def A_skin(self, c, t_skin):
        return self.A(c) - self.A(c - 2.*t_skin)
        
    def x_skin_area_center(self, c, t_skin):
        return (
            self.x_center_area(c)*self.A(c)
            - (t_skin + self.x_center_area(c - 2.*t_skin))*self.A(c - 2.*t_skin)
        )/self.A_skin(c, t_skin)

    def Ixx(self, c):
        return c*c*c*c*self._unit_chord_Ixx

    def Iyy(self, c):
        return c*c*c*c*self._unit_chord_Iyy
    
    def Iyy_skin(self, c, t_skin):
        # Get parameters of outer and inner to establish
        c_i = c - 2.*t_skin
        x_o = self.x_center_area(c)
        x_i = self.x_center_area(c_i) + t_skin
        x_s = self.x_skin_area_center(c, t_skin)
        A_o = self.A(c)
        A_i = self.A(c_i)
        A_s = self.A_skin(c, t_skin)

        # Find moment of inertia around the tip
        Iyy_0 = self.Iyy(c) + x_o*x_o*A_o - self.Iyy(c_i) - x_i*x_i*A_i

        # Return Corrected
        return Iyy_0 - x_s*x_s*A_s

    def Ixx(self, c):
        return c*c*c*c*self._unit_chord_Ixx

    def Ixx_skin(self, c, t_skin):
        return self.Ixx(c) - self.Ixx(c - 2.*t_skin)

    def J(self, c):
        return c*c*c*c*self._unit_chord_J

    def J_skin(self, c, t_skin):
        c_i = c - 2.*t_skin
        x_o = self.x_area_center(c)
        x_i = self.x_area_center(c_i) + t_skin
        x_s = self.x_skin_area_center(c, t_skin)

        return (
            self.J(c) + self.A(c)*np.power(x_o - x_s, 2.)
            - self.J(c_i) - self.A(c_i)*np.power(x_i - x_s, 2.)
        )

    def t_max(self, c):
        return self._t*c

    def curvature(self, x, c):
        return c*self.curvature_unit_chord(x, self._t)