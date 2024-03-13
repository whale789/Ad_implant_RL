import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

class EyeTrackingDensityCalculator:
    def __init__(self, eye_data_path, region_center, region_size):
        self.eye_data = np.loadtxt(eye_data_path, delimiter=',', skiprows=1, encoding="utf-8")
        self.region_center = region_center
        self.region_size = region_size

    def calculate_density(self):
        # Extract eye positions
        eye_positions = self.eye_data[:, [8, 9]]

        # Define the region bounds
        region_bounds = [
            self.region_center[0] - self.region_size[0] / 2,
            self.region_center[0] + self.region_size[0] / 2,
            self.region_center[1] - self.region_size[1] / 2,
            self.region_center[1] + self.region_size[1] / 2
        ]

        # Filter eye positions within the defined region
        region_data = eye_positions[
            (eye_positions[:, 0] >= region_bounds[0]) & (eye_positions[:, 0] <= region_bounds[1]) &
            (eye_positions[:, 1] >= region_bounds[2]) & (eye_positions[:, 1] <= region_bounds[3])
        ]

        if region_data.size == 0:
            print("Eye tracking density in the defined region is 0.")
            return 0

        # Apply kernel density estimation
        kde = KernelDensity(bandwidth=0.1)
        kde.fit(region_data)

        # Generate a grid for density estimation
        x_vals = np.linspace(region_bounds[0], region_bounds[1], 100)
        y_vals = np.linspace(region_bounds[2], region_bounds[3], 100)
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

        # Calculate density values
        density_map = np.exp(kde.score_samples(grid_points)).reshape(100, 100)

        # Visualize the density map
        plt.contourf(x_grid, y_grid, density_map, cmap='viridis', levels=20)
        plt.scatter(region_data[:, 0], region_data[:, 1], c='red', s=10, edgecolor='black')
        plt.title('Eye Tracking Density in Defined Region')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.show()

        # Calculate average density
        average_density = np.mean(np.exp(kde.score_samples(region_data)))

        print(f"Average Eye Tracking Density in Defined Region: {average_density}")
        return round(average_density, 12)

# Example usage
eye_data_path = 'Datas/VR_frame_50/RL_023_50frame.txt'  # Replace with the actual path to your eye tracking data file
region_center = (0.5, 0.5)  # Example center of the region
region_size = (0.2, 0.2)  # Example size of the region

calculator = EyeTrackingDensityCalculator(eye_data_path, region_center, region_size)
calculator.calculate_density()