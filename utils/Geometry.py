import math
from utils.Point import Point

class Geometry:

    @staticmethod
    def modularize(x, mod) -> float:
        """Make a value modular between 0 and mod"""
        if not (-mod <= x < mod):
            if isinstance(x, float):
                x = math.fmod(x, mod)
            else:
                x %= mod

        if x < 0:
            x += mod

        return x

    @staticmethod
    def normalize_angle(value, center=0, amplitude=math.pi) -> float:
        value = value % (2 * amplitude)
        if value < -amplitude + center:
            value += 2 * amplitude
        elif value > amplitude + center:
            value -= 2 * amplitude
        return value
    
    @staticmethod
    def dist_to(p_1: Point, p_2: Point) -> float:
        """Returns the distance between two points"""
        return ((p_1.x - p_2.x) ** 2 + (p_1.y - p_2.y) ** 2) ** 0.5

    @staticmethod
    def smallest_angle_diff(angle_a: float, angle_b: float) -> float:
        """Returns the smallest angle difference between two angles"""
        angle: float = Geometry.modularize(angle_b - angle_a, 2 * math.pi)
        if angle >= math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi
        return angle

    @staticmethod
    def abs_smallest_angle_diff(angle_a: float, angle_b: float) -> float:
        """Returns the absolute smallest angle difference between two angles"""
        return abs(Geometry.smallest_angle_diff(angle_a, angle_b))
    
    @staticmethod
    def from_polar(length: float, angle: float) -> Point:
        return Point(math.cos(angle) * length, math.sin(angle) * length)