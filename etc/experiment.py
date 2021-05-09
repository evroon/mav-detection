import numpy as np

FoV = 47.6 # degrees
drone_size = 1.5 # meters
trajectory_angle = np.deg2rad(20)
trajectory_length = 100
img_width = 2048

observed_angle = np.rad2deg(np.arctan(np.cos(np.pi/2-trajectory_angle)/(1+np.sin(np.pi/2-trajectory_angle))))
observed_drone_size_deg = np.rad2deg(np.arctan(drone_size / (trajectory_length / 10 * 2)))
observed_drone_size_pixels = np.rad2deg(np.arctan(drone_size / trajectory_length)) / FoV * img_width

drone2_start = np.array([np.sin(trajectory_angle), np.cos(trajectory_angle)]) * trajectory_length / 10
drone2_end = np.array([np.sin(trajectory_angle), np.cos(trajectory_angle)]) * trajectory_length
drone2_start_str = ', '.join([f'{x:.03f}' for x in drone2_start])
drone2_end_str = ', '.join([f'{x:.03f}' for x in drone2_end])

print(f'observed trajectory angle: {observed_angle:.03f}')
print(f'radial margin in degrees: {(FoV - observed_angle*2):.03f}')
print(f'observed drone size in degrees: {observed_drone_size_deg:.03f}')
print(f'drone2_start: ({drone2_start_str})')
print(f'drone2_end: ({drone2_end_str})')
print(f'observed drone size for max distance and max resolution: {observed_drone_size_pixels:.00f}')
