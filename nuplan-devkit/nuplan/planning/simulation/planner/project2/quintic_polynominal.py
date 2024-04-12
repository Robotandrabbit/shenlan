import numpy as np
import matplotlib.pyplot as plt

class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, s:float) -> None:
        self.s = s
        self.a0 = xs
        self.a1 = vxs
        self.a2 = 0.5 * axs

        A = np.array([[s ** 3, s ** 4, s ** 5],
                     [3 * s ** 2, 4 * s ** 3, 5 * s ** 4],
                     [6 * s, 12 * s ** 2, 20 * s ** 3]])
        b = np.array([xe - self.a0 - self.a1 * s - self.a2 * s ** 2,
                     vxe - self.a1 - 2 * self.a2 * s,
                     axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def get_param(self)->float:
        return self.s
    
    def get_type(self):
        return "QuinticPolynominal"
    
    def get_point(self, s:float):
        res = 0.0
        if (s <= self.s):
            res = self.a0 + self.a1 * s + self.a2 * s ** 2 + self.a3 * s ** 3 + \
                  self.a4 * s ** 4 + self.a5 * s ** 5
        else:
            x_s = self.a0 + self.a1 * self.s + self.a2 * self.s ** 2 + self.a3 * self.s ** 3 + \
                  self.a4 * self.s ** 4 + self.a5 * self.s ** 5
            delta_s = s - self.s
            v = self.get_first_derivative(self.s)
            a = self.get_second_derivative(self.s)
            res = x_s + v * delta_s + 0.5 * a * delta_s ** 2
        return res

    def get_first_derivative(self, s:float):
        vt = self.a1 + 2 * self.a2 * s + 3 * self.a3 * s ** 2 +\
             4 * self.a4 * s ** 3 + 5 * self.a5 * s ** 4
        return vt
    
    def get_second_derivative(self, s:float):
        at = 2 * self.a2 + 6 * self.a3 * s + 12 * self.a4 * s ** 2 + 20 * self.a5 * s ** 3
        return at

    def get_third_derivative(self, s:float):
        jt = 6 * self.a3 + 24 * self.a4 * s + 60 * self.a5 * s ** 2
        return jt

def main():
    xs = 0.48
    vxs = -0.40
    axs = -0.0
    xe = 1.0
    vxe = 0.0
    axe = 0.0
    s = 26.58

    
    curve = QuinticPolynomial(xs, vxs, axs, xe, vxe, axe, s)
    slist, l = [], []
    for s in np.arange(0.0, 30, 0.1):
        slist.append(s)
        l.append(curve.get_point(s))

    plt.plot(slist, l)
    plt.axis("equal")
    plt.show()

if __name__ == '__main__':
      main()