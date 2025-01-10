import numpy as np
from gymnasium.spaces import Box
from rsoccer_gym.Entities import Ball, Frame, Robot
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree
from utils.Point import Point
from utils.FixedQueue import FixedQueue
from utils.ssl.small_field import SSLHRenderField
from agent import ExampleAgent
from random_agent import RandomAgent
import random
import pygame
from utils.CLI import Difficulty

class SSLExampleEnv(SSLBaseEnv):
    def __init__(self, render_mode=None, difficulty=Difficulty.EASY):
        field = 2
        super().__init__(
            field_type=field,
            n_robots_blue=11,
            n_robots_yellow=11,
            time_step=0.025,
            render_mode=render_mode)
        
        self.DYNAMIC_OBSTACLES, self.max_targets, self.max_rounds = Difficulty.parse(difficulty)

        # Calculate observation space size
        # Main robot: 6 values [x, y, theta, v_x, v_y, v_theta]
        # Other blue robots: 10 robots × 3 values [x, y, theta]
        # Yellow robots: 11 robots × 3 values [x, y, theta]
        # Target: 2 values [x, y]
        n_obs = 6 + (10 * 3) + (11 * 3) + 2  # Total: 69 dimensions

        # Create observation space bounds
        low = np.array([
            # Main robot
            -self.field.length/2,  # x
            -self.field.width/2,   # y
            -np.pi,                # theta
            -self.max_v,           # v_x
            -self.max_v,           # v_y
            -self.max_w,           # v_theta
            
            # Other blue robots (10 robots)
            *[-self.field.length/2] * 10,  # x positions
            *[-self.field.width/2] * 10,   # y positions
            *[-np.pi] * 10,                # thetas
            
            # Yellow robots (11 robots)
            *[-self.field.length/2] * 11,  # x positions
            *[-self.field.width/2] * 11,   # y positions
            *[-np.pi] * 11,                # thetas
            
            # Target
            -self.field.length/2,  # target x
            -self.field.width/2,   # target y
        ])
        
        high = np.array([
            # Main robot
            self.field.length/2,   # x
            self.field.width/2,    # y
            np.pi,                 # theta
            self.max_v,            # v_x
            self.max_v,            # v_y
            self.max_w,            # v_theta
            
            # Other blue robots (10 robots)
            *[self.field.length/2] * 10,   # x positions
            *[self.field.width/2] * 10,    # y positions
            *[np.pi] * 10,                 # thetas
            
            # Yellow robots (11 robots)
            *[self.field.length/2] * 11,   # x positions
            *[self.field.width/2] * 11,    # y positions
            *[np.pi] * 11,                 # thetas
            
            # Target
            self.field.length/2,   # target x
            self.field.width/2,    # target y
        ])

        self.observation_space = Box(
            low=low,
            high=high,
            dtype=np.float32
        )
        
        # Action space remains 2D for velocity and steering
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        self.targets = []
        self.min_dist = 0.18
        self.all_points = FixedQueue(max(4, self.max_targets))
        self.robots_paths = [FixedQueue(40) for i in range(11)]
        
        self.rounds = self.max_rounds
        self.targets_per_round = 1
        
        self.my_agents = {0: ExampleAgent(0, False)}
        self.blue_agents = {i: RandomAgent(i, False) for i in range(1, 11)}
        self.yellow_agents = {i: RandomAgent(i, True) for i in range(0, 11)}
        
        self.gen_target_prob = 0.003
        
        if field == 2:
            self.field_renderer = SSLHRenderField()
            self.window_size = self.field_renderer.window_size

    def _frame_to_observations(self):
        main_robot = self.frame.robots_blue[0]  # Main robot we're controlling
        
        # Find nearest target if any exist
        target_x, target_y = 0, 0
        if self.targets:
            target = self.targets[0]
            target_x, target_y = target.x, target.y
        
        # Create observation array
        observation = []
        
        # Add main robot data
        observation.extend([
            main_robot.x,
            main_robot.y,
            np.deg2rad(main_robot.theta),
            main_robot.v_x,
            main_robot.v_y,
            main_robot.v_theta,
        ])
        
        # Add other blue robots data (excluding main robot)
        for i in range(1, 11):
            robot = self.frame.robots_blue.get(i)
            if robot:
                observation.extend([
                    robot.x,
                    robot.y,
                    np.deg2rad(robot.theta),
                ])
            else:
                observation.extend([0, 0, 0])  # Default values for missing robots
        
        # Add yellow robots data
        for i in range(11):
            robot = self.frame.robots_yellow.get(i)
            if robot:
                observation.extend([
                    robot.x,
                    robot.y,
                    np.deg2rad(robot.theta),
                ])
            else:
                observation.extend([0, 0, 0])  # Default values for missing robots
        
        # Add target position
        observation.extend([target_x, target_y])
        
        return np.array(observation, dtype=np.float32)

    def convert_actions(self, action, angle):
        """Denormalize and convert actions to robot commands"""
        # Denormalize actions
        v_linear = float(action[0]) * self.max_v  # Forward velocity
        v_theta = float(action[2]) * self.max_w   # Angular velocity
        
        # Convert linear velocity to local coordinates
        v_x = v_linear * np.cos(angle)
        v_y = v_linear * np.sin(angle)
        
        # Clip velocities to maximum values
        v_norm = np.hypot(v_x, v_y)
        if v_norm > self.max_v:
            v_x = v_x * self.max_v / v_norm
            v_y = v_y * self.max_v / v_norm
            
        v_theta = np.clip(v_theta, -self.max_w, self.max_w)
        
        return v_x, v_y, v_theta

    def _calculate_reward_and_done(self):
        reward = 0.0
        done = False
        
        robot = self.frame.robots_blue[0]
        
        # Reward for reaching targets
        if self.targets:
            target = self.targets[0]
            dist_to_target = np.hypot(target.x - robot.x, target.y - robot.y)
            reward -= 0.1 * dist_to_target  # Negative reward based on distance
            
            if dist_to_target < self.min_dist:
                reward += 10.0  # Bonus for reaching target
        
        # Penalty for collisions or near-collisions
        for rbt in self.frame.robots_yellow.values():
            dist = np.hypot(rbt.x - robot.x, rbt.y - robot.y)
            if dist < self.min_dist:
                reward -= 20.0
                done = True
            elif dist < self.min_dist * 2:
                reward -= 5.0  # Penalty for getting too close
        
        # Penalty for collisions with blue robots
        for id, rbt in self.frame.robots_blue.items():
            if id != 0:  # Skip self
                dist = np.hypot(rbt.x - robot.x, rbt.y - robot.y)
                if dist < self.min_dist:
                    reward -= 20.0
                    done = True
                elif dist < self.min_dist * 2:
                    reward -= 5.0
        
        # Reward for good steering behavior
        robot_speed = np.hypot(robot.v_x, robot.v_y)
        if 0.2 < robot_speed < self.max_v * 0.8:  # Encourage smooth movement
            reward += 0.1
        
        return reward, done

    def _get_commands(self, actions):
        # Keep only the last M target points
        for target in self.targets:
            if target not in self.all_points:
                self.all_points.push(target)
                
        # Visible path drawing control
        for i in self.my_agents:
            self.robots_paths[i].push(Point(self.frame.robots_blue[i].x, self.frame.robots_blue[i].y))

        # Check if the robot is close to the target
        for j in range(len(self.targets) - 1, -1, -1):
            for i in self.my_agents:
                if Point(self.frame.robots_blue[i].x, self.frame.robots_blue[i].y).dist_to(self.targets[j]) < self.min_dist:
                    self.targets.pop(j)
                    break
        
        # Handle rounds and targets
        if len(self.targets) == 0:
            self.rounds -= 1

        if self.rounds == 0:
            self.rounds = self.max_rounds
            if self.targets_per_round < self.max_targets:
                self.targets_per_round += 1
                self.blue_agents.pop(len(self.my_agents))
                self.my_agents[len(self.my_agents)] = ExampleAgent(len(self.my_agents), False)

        if len(self.targets) == 0:
            for i in range(self.targets_per_round):
                self.targets.append(Point(self.x(), self.y()))
        
        # Handle obstacles and agents
        obstacles = {id: robot for id, robot in self.frame.robots_blue.items()}
        for i in range(0, self.n_robots_yellow):
            obstacles[i + self.n_robots_blue] = self.frame.robots_yellow[i]
        teammates = {id: self.frame.robots_blue[id] for id in self.my_agents.keys()}

        remove_self = lambda robots, selfId: {id: robot for id, robot in robots.items() if id != selfId}

        myActions = []
        for i in self.my_agents.keys():
            action = self.my_agents[i].step(self.frame.robots_blue[i], remove_self(obstacles, i), teammates, self.targets)
            myActions.append(action)

        others_actions = []
        if self.DYNAMIC_OBSTACLES:
            for i in self.blue_agents.keys():
                random_target = []
                if random.uniform(0.0, 1.0) < self.gen_target_prob:
                    random_target.append(Point(x=self.x(), y=self.y()))
                    
                others_actions.append(self.blue_agents[i].step(self.frame.robots_blue[i], obstacles, dict(), random_target, True))

            for i in self.yellow_agents.keys():
                random_target = []
                if random.uniform(0.0, 1.0) < self.gen_target_prob:
                    random_target.append(Point(x=self.x(), y=self.y()))

                others_actions.append(self.yellow_agents[i].step(self.frame.robots_yellow[i], obstacles, dict(), random_target, True))

        return myActions + others_actions

    def _update_targets_and_paths(self):
        # Keep only the last M target points
        for target in self.targets:
            if target not in self.all_points:
                self.all_points.push(target)
                
        # Update path visualization
        for i in self.my_agents:
            self.robots_paths[i].push(Point(self.frame.robots_blue[i].x, self.frame.robots_blue[i].y))

        # Check if robot reached any targets
        for j in range(len(self.targets) - 1, -1, -1):
            for i in self.my_agents:
                if Point(self.frame.robots_blue[i].x, self.frame.robots_blue[i].y).dist_to(self.targets[j]) < self.min_dist:
                    self.targets.pop(j)
                    break
        
        # Handle round completion
        if len(self.targets) == 0:
            self.rounds -= 1
            
            if self.rounds == 0:
                self.rounds = self.max_rounds
                if self.targets_per_round < self.max_targets:
                    self.targets_per_round += 1
                    self.blue_agents.pop(len(self.my_agents))
                    self.my_agents[len(self.my_agents)] = ExampleAgent(len(self.my_agents), False)
            
            # Generate new targets
            for i in range(self.targets_per_round):
                self.targets.append(Point(self.x(), self.y()))
    def x(self):
        return random.uniform(-self.field.length/2 + self.min_dist, self.field.length/2 - self.min_dist)

    def y(self):
        return random.uniform(-self.field.width/2 + self.min_dist, self.field.width/2 - self.min_dist)
    
    def _get_initial_positions_frame(self):

        def theta():
            return random.uniform(0, 360)
    
        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=self.x(), y=self.y())

        pos_frame.robots_blue[0] = Robot(x=self.x(), y=self.y(), theta=theta())

        self.targets = [Point(x=self.x(), y=self.y())]

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        for i in range(self.n_robots_blue):
            pos = (self.x(), self.y())
            while places.get_nearest(pos)[1] < self.min_dist:
                pos = (self.x(), self.y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())
        

        for i in range(0, self.n_robots_yellow):
            pos = (self.x(), self.y())
            while places.get_nearest(pos)[1] < self.min_dist:
                pos = (self.x(), self.y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame
    

    def _render(self):
        def pos_transform(pos_x, pos_y):
            return (
                int(pos_x * self.field_renderer.scale + self.field_renderer.center_x),
                int(pos_y * self.field_renderer.scale + self.field_renderer.center_y),
            )

        super()._render()
        
        for target in self.targets:
            self.draw_target(
                self.window_surface,
                pos_transform,
                target,
                (255, 0, 255),
            )

        if len(self.all_points) > 0:
            my_path = [pos_transform(*p) for p in self.all_points]
            for point in my_path:
                pygame.draw.circle(self.window_surface, (255, 0, 0), point, 3)
        
        for i in range(len(self.robots_paths)):
            if len(self.robots_paths[i]) > 1:
                my_path = [pos_transform(*p) for p in self.robots_paths[i]]
                pygame.draw.lines(self.window_surface, (255, 0, 0), False, my_path, 1)
        
         # Add A* path rendering
        for agent_id, agent in self.my_agents.items():
            if hasattr(agent, 'get_planned_path'):
                planned_path = agent.get_planned_path()
                if planned_path:
                    # Convert path points to screen coordinates
                    path_points = [pos_transform(p.x, p.y) for p in planned_path]
                    # Draw planned path in yellow
                    pygame.draw.lines(self.window_surface, (255, 255, 0), False, path_points, 2)
                    # Draw waypoints
                    for point in path_points:
                        pygame.draw.circle(self.window_surface, (255, 255, 0), point, 4)

    def draw_target(self, screen, transformer, point, color):
        x, y = transformer(point.x, point.y)
        size = 0.09 * self.field_renderer.scale
        pygame.draw.circle(screen, color, (x, y), size, 2)