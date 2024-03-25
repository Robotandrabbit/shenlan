import numpy as np

class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, time) -> None:
        self.a0 = xs
        self.a1 = vxs
        self.a2 = 0.5 * axs

        A = np.array([time ** 3, time ** 4, time ** 5]
                     [3 * time ** 2, 4 * time ** 3, 5 * time ** 4]
                     [6 * time, 12 * time ** 2, 20 * time ** 3])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2]
                     [vxe - self.a1 - 2 * self.a2 * time]
                     [axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]
      
    def get_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3 + \
             self.a4 * t ** 4 + self.a5 * t ** 5
        return xt

    def get_first_derivative(self, t):
        vt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 +\
             4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4
        return vt
    
    def get_second_derivative(self, t):
        at = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3
        return at

    def get_third_derivative(self, t):
        jt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2
        return jt