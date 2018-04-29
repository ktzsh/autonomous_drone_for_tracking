import os
import sys
import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image

python_path = os.path.abspath('TF_ObjectDetection')
sys.path.append(python_path)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from object_detection.utils import visualization_utils as vis_util

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
        The (x1, y1) position is at the top left corner,
        The (x2, y2) position is at the bottom right corner
        Cartesian Co-ordinate System with origin at center of frame right and top are positive axis
    """

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] > bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] > bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left   = max(bb1['x1'], bb2['x1'])
    y_top    = min(bb1['y1'], bb2['y1'])
    x_right  = min(bb1['x2'], bb2['x2'])
    y_bottom = max(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom > y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def nearest_to_step(dist, step_sizes):
    best_step = 0
    step = abs(step_sizes[0] - abs(dist))
    for i, step_size in enumerate(step_sizes[1:]):
        step_t = abs(step_size - abs(dist))
        if step_t<=step:
            step = step_t
            best_step = i + 1

    if dist<0:
        return -1*step_sizes[best_step]
    else:
        return step_sizes[best_step]


step_sizes = [0, 10, 40]
for dir in ['seq4', 'seq5', 'seq6', 'seq7', 'seq8']:
    iou1 = 0.0
    iou2 = 0.0

    if not dir[:3]=="seq":
        continue
    for _, _, files in os.walk("data/"+dir):
        frame_id = 1
        bbt = None
        bb1 = None
        files.sort()
        for f in files:
            if not f.endswith(".txt"):
                continue
            file = "data/"+dir+"/"+f

            img     = np.asarray(Image.open(file.split('.')[0] + '.png').convert('RGB'), dtype=np.uint8)
            img_rgb = img.copy()

            tree = ET.parse(file.split('.')[0] + '.xml')
            root = tree.getroot()
            obj = root.findall('object')
            bndbox = obj[0].find('bndbox')

            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)

            vis_util.draw_bounding_boxes_on_image_array( img_rgb,
                                                         np.array([[(float(ymin)/720.0),
                                                                    (float(xmin)/1280.0),
                                                                    (float(ymax)/720.0),
                                                                    (float(xmax)/1280.0)]]),
                                                         color='black',
                                                         thickness=4)
            bb2 = {
                    'x1': xmin - 1280/2,
                    'x2': xmax - 1280/2,
                    'y1': 720/2 - ymin,
                    'y2': 720/2 - ymax
            }

            with open(file, 'rb') as f:
                xmin, xmax, ymin, ymax = [int(float(x)  ) for x in f.readline().split()]
                vis_util.draw_bounding_boxes_on_image_array( img_rgb,
                                                             np.array([[(float(ymin)/720.0),
                                                                        (float(xmin)/1280.0),
                                                                        (float(ymax)/720.0),
                                                                        (float(xmax)/1280.0)]]),
                                                             color='red',
                                                             thickness=4)
                bbt = {
                        'x1': xmin - 1280/2,
                        'x2': xmax - 1280/2,
                        'y1': 720/2 - ymin ,
                        'y2': 720/2 - ymax
                }

            if frame_id==1:
                bb1 = bbt
            else:
                cxt = int(float(bbt['x2'] + bbt['x1'])/2.0)
                cx1 = int(float(bb1['x2'] + bb1['x1'])/2.0)
                cyt = int(float(bbt['y2'] + bbt['y1'])/2.0)
                cy1 = int(float(bb1['y2'] + bb1['y1'])/2.0)

                dx = cxt - cx1
                dy = cyt - cy1
                print dx, dy

                step_x = nearest_to_step(dx, step_sizes)
                step_y = nearest_to_step(dy, step_sizes)
                print step_x, step_y

                cx = cx1 + step_x
                cy = cy1 + step_y
                h  = float(bbt['y1'] - bbt['y2'])
                w  = float(bbt['x2'] - bbt['x1'])
                bb1 = {
                        'x1': int(cx - w/2.0),
                        'x2': int(cx + w/2.0),
                        'y1': int(cy + h/2.0),
                        'y2': int(cy - h/2.0)
                }
                xmin = bb1['x1'] + 1280/2
                xmax = bb1['x2'] + 1280/2
                ymin = 720/2 - bb1['y1']
                ymax = 720/2 - bb1['y2']
                vis_util.draw_bounding_boxes_on_image_array( img_rgb,
                                                             np.array([[(float(ymin)/720.0),
                                                                        (float(xmin)/1280.0),
                                                                        (float(ymax)/720.0),
                                                                        (float(xmax)/1280.0)]]),
                                                             color='blue',
                                                             thickness=4)


            print bb1
            iou_t1 = get_iou(bb2, bbt)
            iou_t2 = get_iou(bb2, bb1)
            print iou_t1, iou_t2
            iou1 += iou_t1
            iou2 += iou_t2

            frame_id += 1

            import cv2

            cv2.imshow("!", img_rgb)
            cv2.waitKey(10)

    print "\n---Sequence:", dir, "---"
    print "IoU b/w GT and Tracker(Unconstrained)         :", iou1
    print "IoU b/w GT and Tracker(Constrained Baseline)  :", iou2
