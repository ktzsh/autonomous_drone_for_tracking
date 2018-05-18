
import os
import sys
import time

python_path = os.path.abspath('AirSim/PythonClient')
sys.path.append(python_path)
from AirSimClient import *

client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

INITZ = -19
client.takeoff()
client.moveToPosition(0, 0, INITZ, 5)
time.sleep(5)

count = 3000
print "BEGIN"
while(True):
    path = 'data/detector/' + str(count).zfill(6) + '.png'
    response = client.simGetImages([ImageRequest(3, AirSimImageType.Scene, False, False)])[0]
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) # get numpy array
    img_rgba = img1d.reshape(response.height, response.width, 4) # reshape array to 4 channel image array H X W X 4
    img_rgba = np.flipud(img_rgba) # original image is fliped vertically
    client.write_png(os.path.normpath(path), img_rgba)

    count += 1
    if count%400==0:
        print count, INITZ
        INITZ -= 3
        client.moveToPosition(0, 0, INITZ, 5)
        time.sleep(5)
