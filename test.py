import time
# from Environment import Environment
from MultiRotorConnector import MultiRotorConnector

# env = Environment()
# while True:
#     current_state = env.reset()
#     time.sleep(5)

connector = MultiRotorConnector()
time.sleep(4)
_ = connector.get_frame(path='1.png')
position = connector.get_position()
x = position.x_val
y = position.y_val
z = position.z_val
print x, y, z

connector.move_to_position([x, y, z])
time.sleep(4)
_ = connector.get_frame(path='1.png')
position = connector.get_position()
x = position.x_val
y = position.y_val
z = position.z_val
print x, y, z

connector.move_to_position([x, y, z+5])
time.sleep(4)
_ = connector.get_frame(path='2.png')
position = connector.get_position()
x = position.x_val
y = position.y_val
z = position.z_val
print x, y, z

connector.move_to_position([x, y, z+5])
time.sleep(4)
_ = connector.get_frame(path='3.png')
position = connector.get_position()
x = position.x_val
y = position.y_val
z = position.z_val
print x, y, z

connector.move_to_position([x, y, z-15])
time.sleep(4)
_ = connector.get_frame(path='4.png')
position = connector.get_position()
x = position.x_val
y = position.y_val
z = position.z_val
print x, y, z

connector.move_to_position([x, y, z-5])
time.sleep(4)
_ = connector.get_frame(path='5.png')
position = connector.get_position()
x = position.x_val
y = position.y_val
z = position.z_val
print x, y, z
