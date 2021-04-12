import cv2
import airsim
import numpy as np

client = airsim.MultirotorClient()
client.confirmConnection()
print(client.getMultirotorState().kinematics_estimated.position)

responses = client.simGetImages([
    airsim.ImageRequest("depth", airsim.ImageType.DepthPerspective, True, True),
], vehicle_name='Drone1')

pfm_array = airsim.get_pfm_array(responses[0])
pfm_array_int = (pfm_array / np.max(pfm_array) * 255) * 5
pfm_array_int = np.clip(0, 255, pfm_array_int).astype(np.uint8)
pfm_array_int = cv2.applyColorMap(pfm_array_int, cv2.COLORMAP_JET)
cv2.imwrite('test.png', pfm_array_int)
