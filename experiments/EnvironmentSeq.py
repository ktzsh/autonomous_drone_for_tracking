import os
import sys
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import xml.etree.ElementTree as ET

python_path = os.path.abspath('TF_ObjectDetection')
sys.path.append(python_path)
from object_detection.utils import visualization_utils as vis_util

# from Detector import Detector

class State():
    DELTA_X    = 0.0
    DELTA_Y    = 0.0
    pass

class EnvironmentSeq:
    def __init__(self, image_shape=(720, 1280), step_sizes=[-40, -20, 0, 20, 40], max_guided_eps=1000):
        self.ncols = 45
        self.nrows = 45

        self.current_episode = 0
        self.max_guided_eps  = max_guided_eps

        self.im_width  = image_shape[1]
        self.im_height = image_shape[0]

        self.step_sizes = step_sizes

        self.fig        = None
        self.max_dist   = np.linalg.norm([self.im_width, self.im_height])
        self.SCALE_IOU  = 8.0
        self.SCALE_DIST = 1.0

        self.current_sequence = 0
        self.current_frame    = 0
        self.length_sequences = []
        self.frames_sequences = []

        self._sequences = ['seq4', 'seq5', 'seq6', 'seq7', 'seq8']
        for dir in self._sequences:
            frames = [f for f in os.listdir('data/'+dir) if f.endswith('.txt')]
            frames.sort()
            self.length_sequences += [len(frames)]
            self.frames_sequences += [frames]

        self.old_x = None
        self.old_y = None

    def state_to_array(self, state):
        out = np.zeros((2,), dtype='float32')
        out[0] = float(state.DELTA_X)/float(self.im_width)
        out[1] = float(state.DELTA_Y)/float(self.im_height)
        return out

    def nearest_to_step(self, dist):
        nearest_dist = [abs(step_size - dist) for step_size in self.step_sizes]
        return self.step_sizes[nearest_dist.index(min(nearest_dist))]

    def reset(self):
        self.current_episode += 1

        self.current_frame = 0
        file    = 'data/' + str(self._sequences[self.current_sequence]) + '/' + \
                  str(self.frames_sequences[self.current_sequence][self.current_frame])
        frame   = np.asarray(Image.open(file.split('.')[0] + '.png').convert('RGB'), dtype=np.uint8).copy()
        det_box = None
        with open(file.split('.')[0] + '.txt', 'rb') as f:
            xmint, xmaxt, ymint, ymaxt = [int(float(x)) for x in f.readline().split()]
            det_box = {
                    'x1': xmint - self.im_width/2,
                    'x2': xmaxt - self.im_width/2,
                    'y1': self.im_height/2 - ymint,
                    'y2': self.im_height/2 - ymaxt
            }
        POS_X1  = int(float(det_box['x1'] + det_box['x2'])/2.0)
        POS_Y1  = int(float(det_box['y1'] + det_box['y2'])/2.0)
        WIDTH1  = int(float(det_box['x2'] - det_box['x1']))
        HEIGHT1 = int(float(det_box['y1'] - det_box['y2']))


        self.current_frame = 1
        file    = 'data/' + str(self._sequences[self.current_sequence]) + '/' + \
                  str(self.frames_sequences[self.current_sequence][self.current_frame])
        frame   = np.asarray(Image.open(file.split('.')[0] + '.png').convert('RGB'), dtype=np.uint8).copy()
        det_box = None
        with open(file.split('.')[0] + '.txt', 'rb') as f:
            xmint, xmaxt, ymint, ymaxt = [int(float(x)) for x in f.readline().split()]
            det_box = {
                    'x1': xmint - self.im_width/2,
                    'x2': xmaxt - self.im_width/2,
                    'y1': self.im_height/2 - ymint,
                    'y2': self.im_height/2 - ymaxt
            }
        POS_X2  = int(float(det_box['x1'] + det_box['x2'])/2.0)
        POS_Y2  = int(float(det_box['y1'] + det_box['y2'])/2.0)
        WIDTH2  = int(float(det_box['x2'] - det_box['x1']))
        HEIGHT2 = int(float(det_box['y1'] - det_box['y2']))

        _state = State()
        _state.DELTA_X = POS_X2 - POS_X1
        _state.DELTA_Y = POS_Y2 - POS_Y1
        print "\n-----Parameters-----"
        print "Delta    X:", _state.DELTA_X
        print "Delta    Y:", _state.DELTA_Y

        self.old_x          = POS_X1
        self.old_y          = POS_Y1
        self.old_gx         = POS_X1
        self.old_gy         = POS_Y1
        self.current_state  = _state

        return self.state_to_array(_state)

    def get_iou(self, bb1, bb2):
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] > bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] > bb2['y2']

        x_left   = max(bb1['x1'], bb2['x1'])
        y_top    = min(bb1['y1'], bb2['y1'])
        x_right  = min(bb1['x2'], bb2['x2'])
        y_bottom = max(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom > y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def step(self, action):
        _state = State()

        file   = 'data/' + str(self._sequences[self.current_sequence]) + '/' + \
                 str(self.frames_sequences[self.current_sequence][self.current_frame])
        frame  = np.asarray(Image.open(file.split('.')[0] + '.png').convert('RGB'), dtype=np.uint8).copy()

        # GROUNDTRUTH
        tree   = ET.parse(file.split('.')[0]+'.xml')
        root   = tree.getroot()
        obj    = root.findall('object')
        bndbox = obj[0].find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        gt_box = {
                    'x1': xmin - self.im_width/2,
                    'x2': xmax - self.im_width/2,
                    'y1': self.im_height/2 - ymin,
                    'y2': self.im_height/2 - ymax
        }

        # TRACKER
        det_box = None
        with open(file, 'rb') as f:
            xmint, xmaxt, ymint, ymaxt = [int(float(x)) for x in f.readline().split()]
            det_box = {
                    'x1': xmint - self.im_width/2,
                    'x2': xmaxt - self.im_width/2,
                    'y1': self.im_height/2 - ymint,
                    'y2': self.im_height/2 - ymaxt
            }
        POS_X  = int(float(det_box['x1'] + det_box['x2'])/2.0)
        POS_Y  = int(float(det_box['y1'] + det_box['y2'])/2.0)
        WIDTH  = int(float(det_box['x2'] - det_box['x1']))
        HEIGHT = int(float(det_box['y1'] - det_box['y2']))

        # GREEDY (BASELINE) AGENT
        step_x = self.nearest_to_step(int(float(gt_box['x1'] + gt_box['x2'])/2.0) - self.old_gx)
        step_y = self.nearest_to_step(int(float(gt_box['y1'] + gt_box['y2'])/2.0)- self.old_gy)
        new_gx  = self.old_gx + step_x
        new_gy  = self.old_gy + step_y
        baseline_box = {
                    'x1': new_gx - WIDTH/2,
                    'x2': new_gx + WIDTH/2,
                    'y1': new_gy + HEIGHT/2,
                    'y2': new_gy - HEIGHT/2
        }

        # CUSTOM AGENT
        print "\n-----Parameters-----"
        print "Delta    X:", self.current_state.DELTA_X
        print "Delta    Y:", self.current_state.DELTA_Y
        new_x      = self.old_x + action[0]
        new_y      = self.old_y + action[1]
        agent_box = {
                    'x1': new_x - WIDTH/2,
                    'x2': new_x + WIDTH/2,
                    'y1': new_y + HEIGHT/2,
                    'y2': new_y - HEIGHT/2
        }

        if self.current_episode>self.max_guided_eps:
            self.old_x  = new_x # POS_X
            self.old_y  = new_y # POS_Y
        else:
            self.old_x  = POS_X
            self.old_y  = POS_Y
        self.old_gx = new_gx
        self.old_gy = new_gy

        iou_unconstrained = self.get_iou(gt_box, det_box)
        iou_constrained   = self.get_iou(gt_box, agent_box)
        iou_baseline      = self.get_iou(gt_box, baseline_box)
        print "-----IoU Stats-----"
        print "Uncons IoU:", iou_unconstrained
        print "Cons   IoU:", iou_constrained
        print "Greedy IoU:", iou_baseline

        dist_x = float(agent_box['x1'] + agent_box['x2'])/2.0 - float(gt_box['x1'] + gt_box['x2'])/2.0
        dist_y = float(agent_box['y1'] + agent_box['y2'])/2.0 - float(gt_box['y1'] + gt_box['y2'])/2.0
        dist = np.linalg.norm([dist_x, dist_y])/self.max_dist
        reward = ((1-dist)*self.SCALE_DIST + iou_constrained*self.SCALE_IOU)/(self.SCALE_DIST + self.SCALE_IOU)
        print "Distance   :", dist, \
            "\nIoU        :", iou_constrained, \
            "\nDist Reward:", (1-dist)*self.SCALE_DIST, \
            "\nIoU Reward :", iou_constrained*self.SCALE_IOU

        vis_util.draw_bounding_boxes_on_image_array( frame,
                                                     np.array([[(float(ymin)/float(self.im_height)),
                                                                (float(xmin)/float(self.im_width)),
                                                                (float(ymax)/float(self.im_height)),
                                                                (float(xmax)/float(self.im_width))]]),
                                                     color='black',
                                                     thickness=7)
        vis_util.draw_bounding_boxes_on_image_array( frame,
                                                     np.array([[(float(self.im_height/2 - det_box['y1'])/float(self.im_height)),
                                                                (float(det_box['x1'] + self.im_width/2)/float(self.im_width)),
                                                                (float(self.im_height/2 - det_box['y2'])/float(self.im_height)),
                                                                (float(det_box['x2'] + self.im_width/2)/float(self.im_width))]]),
                                                     color='blue',
                                                     thickness=3)
        vis_util.draw_bounding_boxes_on_image_array( frame,
                                                     np.array([[(float(self.im_height/2 - agent_box['y1'])/float(self.im_height)),
                                                                (float(agent_box['x1'] + self.im_width/2)/float(self.im_width)),
                                                                (float(self.im_height/2 - agent_box['y2'])/float(self.im_height)),
                                                                (float(agent_box['x2'] + self.im_width/2)/float(self.im_width))]]),
                                                     color='yellow',
                                                     thickness=5)
        vis_util.draw_bounding_boxes_on_image_array( frame,
                                                     np.array([[(float(self.im_height/2 - baseline_box['y1'])/float(self.im_height)),
                                                                (float(baseline_box['x1'] + self.im_width/2)/float(self.im_width)),
                                                                (float(self.im_height/2 - baseline_box['y2'])/float(self.im_height)),
                                                                (float(baseline_box['x2'] + self.im_width/2)/float(self.im_width))]]),
                                                     color='red',
                                                     thickness=5)

        result = Image.fromarray(frame)
        path_prefix = 'data/output_train/' + self._sequences[self.current_sequence] + '/'
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)
        img_path = path_prefix + self.frames_sequences[self.current_sequence][self.current_frame].split('.')[0] + '.jpg'
        result.save(img_path)

        done = 0
        self.current_frame += 1
        if self.current_frame>=self.length_sequences[self.current_sequence]:
            done = 1
            self.current_sequence =  (self.current_sequence + 1)%len(self._sequences)
        print "Action RL  :", action, "pixels"
        print "Action BASE:", (step_x, step_y), "pixels"
        print "Reward     :", reward
        print "Done       :", done

        # cv2.imshow('Simulation', frame)
        # cv2.waitKey(10)

        _state = State()
        if not done:
            file   = 'data/' + str(self._sequences[self.current_sequence]) + '/' + \
                     str(self.frames_sequences[self.current_sequence][self.current_frame])
            det_box = None
            with open(file, 'rb') as f:
                xmint, xmaxt, ymint, ymaxt = [int(float(x)) for x in f.readline().split()]
                det_box = {
                        'x1': xmint - self.im_width/2,
                        'x2': xmaxt - self.im_width/2,
                        'y1': self.im_height/2 - ymint,
                        'y2': self.im_height/2 - ymaxt
                }
            POS_X  = int(float(det_box['x1'] + det_box['x2'])/2.0)
            POS_Y  = int(float(det_box['y1'] + det_box['y2'])/2.0)
            WIDTH  = int(float(det_box['x2'] - det_box['x1']))
            HEIGHT = int(float(det_box['y1'] - det_box['y2']))

            _state.DELTA_X = POS_X - self.old_x
            _state.DELTA_Y = POS_Y - self.old_y
            self.current_state = _state

        return self.state_to_array(_state), reward, done
