from math import cos, sin, atan2, sqrt
import numpy as np


class Position(object):
    def __init__(self, x1, x2, x3, c_type):
        assert c_type.lower() in ["cartesian", "polar"]

        self.x, self.y, self.z = 0.0, 0.0, 0.0
        self.phi, self.nu, self.r = 0.0, 0.0, 0.0
        if c_type == "cartesian":
            self.set_cartesian(x1, x2, x3)
        else:
            self.set_polar(x1, x2, x3)

    def clone(self):
        return Position(self.x, self.y, self.z, "cartesian")

    def set_cartesian(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.calc_polar()
        self.calc_cartesian()

    def set_polar(self, phi, nu, r):
        self.phi, self.nu, self.r = float(phi), float(nu), float(r)
        self.calc_cartesian()
        self.calc_polar()

    def calc_cartesian(self):
        self.x = self.r * cos(self.phi) * cos(self.nu)
        self.y = self.r * sin(self.phi) * cos(self.nu)
        self.z = self.r * sin(self.nu)

    def calc_polar(self):
        self.phi = atan2(self.y, self.x)
        self.nu = atan2(self.z, sqrt(self.x**2 + self.y**2))
        self.r = sqrt(self.x**2 + self.y**2 + self.z**2)

    def rotate(self, rot_matrix):
        pos = np.dot(rot_matrix, np.array([self.x, self.y, self.z]).reshape(3, 1))
        self.x, self.y, self.z = pos[0], pos[1], pos[2]
        self.calc_polar()
        self.calc_cartesian()

    def set_radius(self, radius):
        self.r = radius
        self.calc_cartesian()

    def coords(self, c_type):
        if c_type == "cartesian":
            return np.array([self.x, self.y, self.z])
        elif c_type == "polar":
            return np.array([self.phi, self.nu, self.r])
        else:
            raise ValueError("Unknown coordinate type. Use cartesian or polar.")

    def print_position(self, c_type=None):
        if c_type is None or c_type == "cartesian":
            print("Cartesian (x,y,z): (%.2f, %.2f, %.2f)" % (self.x, self.y, self.z))
        if c_type is None or c_type == "polar":
            print("Polar (phi,nu,r):  (%.2f, %.2f, %.2f)" % (self.phi, self.nu, self.r))

