from typing import List, Tuple
import numpy as np

class GridConverter:
    def __init__(self, field_length: float, field_width: float, grid_size: int = 35):
        """
        Initialize grid converter for SSL environment
        
        Args:
            field_length: Length of the field in meters
            field_width: Width of the field in meters
            grid_size: Number of cells in each dimension
        """
        self.field_length = field_length
        self.field_width = field_width
        self.grid_size = grid_size
        
        # Calculate cell dimensions
        self.cell_length = field_length / grid_size
        self.cell_width = field_width / grid_size
        
    def create_grid(self, obstacles: dict, robot_radius: float) -> np.ndarray:
        """
        Create a grid representation of the environment
        
        Args:
            obstacles: Dictionary of robot obstacles {id: robot}
            robot_radius: Radius of robots for collision checking
            
        Returns:
            Binary grid where 1 represents obstacles and 0 represents free space
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # For each cell, check if it intersects with any obstacle
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_x = -self.field_length/2 + i * self.cell_length + self.cell_length/2
                cell_y = -self.field_width/2 + j * self.cell_width + self.cell_width/2
                
                # Check each obstacle
                for robot in obstacles.values():
                    # Calculate distance to robot center
                    dist = np.hypot(robot.x - cell_x, robot.y - cell_y)
                    if dist < robot_radius + self.cell_length/2:  # Add cell radius for conservative estimate
                        grid[i, j] = 1
                        break
                        
        return grid
    
    def continuous_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous coordinates to grid coordinates"""
        grid_x = int((x + self.field_length/2) / self.cell_length)
        grid_y = int((y + self.field_width/2) / self.cell_width)
        
        # Clamp to grid bounds
        grid_x = max(0, min(grid_x, self.grid_size-1))
        grid_y = max(0, min(grid_y, self.grid_size-1))
        
        return grid_x, grid_y
    
    def grid_to_continuous(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to continuous coordinates"""
        x = -self.field_length/2 + grid_x * self.cell_length + self.cell_length/2
        y = -self.field_width/2 + grid_y * self.cell_width + self.cell_width/2
        return x, y