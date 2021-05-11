import numpy as np

FoV = 47.6 # degrees
VFoV = 36.6
drone_width = 1.6 # meters
drone_height = 0.23 # meters
trajectory_angle = np.deg2rad(20)
trajectory_length = 120
img_width = 2048
img_height = 1536
fps = 38

observed_angle = np.rad2deg(np.arctan(np.cos(np.pi/2-trajectory_angle)/(1+np.sin(np.pi/2-trajectory_angle))))
observed_drone_width_deg = np.rad2deg(np.arctan(drone_width / (trajectory_length / 10 * 2)))
observed_drone_width_pixels = np.rad2deg(np.arctan(drone_width / trajectory_length)) / FoV * img_width

drone2_start = np.array([np.sin(trajectory_angle), np.cos(trajectory_angle)]) * trajectory_length / 10
drone2_end = np.array([np.sin(trajectory_angle), np.cos(trajectory_angle)]) * trajectory_length
drone2_start_str = ', '.join([f'{x:.03f}' for x in drone2_start])
drone2_end_str = ', '.join([f'{x:.03f}' for x in drone2_end])

print(f'observed trajectory angle: {observed_angle:.03f}')
print(f'radial margin in degrees: {(FoV - observed_angle*2):.03f}')
print(f'observed drone size in degrees: {observed_drone_width_deg:.03f}')
print(f'drone2_start: ({drone2_start_str})')
print(f'drone2_end: ({drone2_end_str})')
print(f'observed drone size for max distance and max resolution: {observed_drone_width_pixels:.00f}')


pattern_length = 120
cross_length = np.arctan(np.deg2rad(FoV/2)) * pattern_length
observer_velocity = 4
max_distance = np.sqrt(pattern_length ** 2 + cross_length ** 2)
drone_width_deg_min = np.rad2deg(np.arctan(drone_width / 2 / max_distance)) * 2 / FoV * img_width
drone_height_deg_min = np.rad2deg(np.arctan(drone_height / 2 / max_distance)) * 2 / VFoV * img_height
print(f'observed min drone size in pixels: {drone_width_deg_min:.03f}x{drone_height_deg_min:.03f} pixels')
print(f'cross_length: {cross_length:.01f}m')
print()



for velocity in [3, 6, 10, 15]:
    distance_at_crossing = pattern_length - cross_length / velocity * observer_velocity
    drone_width_deg_max = np.rad2deg(np.arctan(drone_width / 2 / distance_at_crossing)) * 2 / FoV * img_width
    drone_height_deg_max = np.rad2deg(np.arctan(drone_height / 2 / distance_at_crossing)) * 2 / VFoV * img_height
    print()
    print(f'distance at crossing: {distance_at_crossing:.03f}m')

    min_flow = np.rad2deg(np.arctan(velocity/pattern_length)) / FoV * img_width / fps * np.sin(np.deg2rad(FoV/2))
    max_flow = np.rad2deg(np.arctan(velocity/distance_at_crossing)) / FoV * img_width / fps

    print(f'min flow {velocity}m/s: {min_flow:.03f} px/dt')
    print(f'max flow {velocity}m/s: {max_flow:.03f} px/dt')

    print(f'observed max drone size in pixels for {velocity}m/s: {drone_width_deg_max:.03f}x{drone_height_deg_max:.03f} pixels')
