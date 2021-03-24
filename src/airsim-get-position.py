import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
print(client.getMultirotorState().kinematics_estimated.position)
