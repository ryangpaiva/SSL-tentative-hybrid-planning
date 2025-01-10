from utils.ssl.Navigation import Navigation
from utils.ssl.base_agent import BaseAgent
from planning.plan import Plan
from planning.grid_converter import GridConverter
from utils.Point import Point
from rsoccer_gym.Entities import Robot

class ExampleAgent(BaseAgent):
    def __init__(self, id=0, yellow=False):
        super().__init__(id, yellow)
        self.grid_converter = GridConverter(field_length=9.0, field_width=6.0)
        self.robot_radius = 0.09
        self.planned_path = []  # Store the planned path

    def step(self, robot, obstacles, teammates, targets):
        self.robot = robot
        self.targets = targets
        self.decision(obstacles)
        self.post_decision()
        return Robot(id=self.id, yellow=self.yellow,
                    v_x=self.next_vel.x, v_y=self.next_vel.y, v_theta=self.angle_vel)

    def decision(self, obstacles):
        if len(self.targets) == 0:
            return

        # Create grid representation of current state
        grid = self.grid_converter.create_grid(obstacles, self.robot_radius)
        
        # Convert start and goal positions to grid coordinates
        start_grid = self.grid_converter.continuous_to_grid(self.robot.x, self.robot.y)
        goal_grid = self.grid_converter.continuous_to_grid(self.targets[0].x, self.targets[0].y)
        
        # Get path using A*
        grid_path = Plan.astar(grid, start_grid, goal_grid)
        
        if grid_path and len(grid_path) > 1:
            # Store the full path in continuous coordinates
            self.planned_path = [
                Point(*self.grid_converter.grid_to_continuous(x, y))
                for x, y in grid_path
            ]
            
            # Use next waypoint for navigation
            next_x, next_y = self.grid_converter.grid_to_continuous(grid_path[1][0], grid_path[1][1])
            next_point = Point(next_x, next_y)
            
            # Use Navigation to move to next waypoint
            target_velocity, target_angle_velocity = Navigation.goToPoint(self.robot, next_point)
            self.set_vel(target_velocity)
            self.set_angle_vel(target_angle_velocity)
        else:
            self.planned_path = []  # Clear path if no valid path found

    def post_decision(self):
        pass

    def get_planned_path(self):
        """Return the current planned path for rendering"""
        return self.planned_path