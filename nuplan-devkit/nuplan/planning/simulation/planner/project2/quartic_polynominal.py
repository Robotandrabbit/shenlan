import numpy as np
import matplotlib.pyplot as plt

class QuarticPolynominal:
  def __init__(self, sxs, vxs, axs, vxe, axe, time) -> None:
      self.time = time
      self.a0 = sxs
      self.a1 = vxs
      self.a2 = 0.5 * axs

      A = np.array([[3 * time ** 2, 4 * time ** 3],
                    [6 * time, 12 * time ** 2]])
      
      b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                    axe - 2 * self.a2])

      solution = np.linalg.solve(A, b)

      self.a3 = solution[0]
      self.a4 = solution[1]

  def get_type(self):
      return "QuarticPolynominal"

  def get_time(self):
      return self.time

  def get_point(self, t) -> float:
      val = self.a0 + self.a1 * t + self.a2 * t ** 2 + self.a3 * t ** 3 + self.a4 * t ** 4
      return val

  def get_first_derivative(self, t) -> float:
      val = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3
      return val
  
  def get_second_derivative(self, t) -> float:
      val = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2
      return val
  
  def get_third_derivative(self, t) -> float:
      val = 6 * self.a3 + 12 * 2 * self.a4 * t
      return val
  

def main():
  sxs = 0.0
  vxs = 12.0
  axs = 0.0
  vxe = 5.0
  axe = 0.0
  time = 2.0
  curve = QuarticPolynominal(sxs, vxs, axs, vxe, axe, time)

  tl = []
  s = []
  v = []
  for t in np.arange(0.0, time, 0.1):
    tl.append(t)
    s.append(curve.get_point(t))
    v.append(curve.get_first_derivative(t))

  plt.figure()
  plt.plot(tl, s)

  plt.figure()
  plt.plot(tl, v)
  plt.show()

if __name__ == '__main__':
  main()