from rsoccer_gym.Entities import Robot
from utils.Point import Point

class BaseAgent:
    """Abstract Agent."""

    def __init__(self, id=0, yellow=False):
        self.id = id
        self.robot = Robot()
        self.pos = Point(0, 0)
        self.vel = Point(0, 0)
        self.body_angle = float(0)
        self.targets = []
        self.yellow = yellow
        self.opponents = dict()
        self.teammates = dict()

        self.next_vel = Point(0, 0)
        self.angle_vel = float(0)

    def step(self, self_robot : Robot, 
             opponents: dict[int, Robot] = dict(), 
             teammates: dict[int, Robot] = dict(), 
             targets: list[Point] = [], 
             keep_targets=False) -> Robot:

        self.reset()
        self.pos = Point(self_robot.x, self_robot.y)
        self.vel = Point(self_robot.v_x, self_robot.v_y)
        self.body_angle = self_robot.theta

        if len(targets) > 0:
            self.targets = targets.copy()
        elif len(self.targets) == 0 or not keep_targets:
            self.targets = []
            
        self.robot = self_robot
        self.opponents = opponents.copy()
        self.teammates = teammates.copy()

        self.decision()
        self.post_decision()

        return Robot( id=self.id, yellow=self.yellow,
                      v_x=self.next_vel.x, v_y=self.next_vel.y, v_theta=self.angle_vel)

    def reset(self):
        self.next_vel = Point(0, 0)
        self.angle_vel = 0

    def decision(self):
        raise NotImplementedError()
    
    def post_decision(self):
        raise NotImplementedError()
    
    def set_vel(self, vel: Point):
        self.next_vel = vel
    
    def set_angle_vel(self, angle_vel: float):
        self.angle_vel = angle_vel