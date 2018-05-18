import os
import sys
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
    def __init__(self):
        self.ncols = 45
        self.nrows = 45

        self.fig        = None
        self.max_dist   = 735.0
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

        # self._detector = Detector()

        self.old_x = None
        self.old_y = None

    def state_to_array(self, state):
        out = np.zeros((2,), dtype='float32')
        out[0] = state.DELTA_X
        out[1] = state.DELTA_Y
        return out

    def reset(self):
        self.current_frame = 0
        file    = 'data/' + str(self._sequences[self.current_sequence]) + '/' + \
                  str(self.frames_sequences[self.current_sequence][self.current_frame])
        frame   = np.asarray(Image.open(file.split('.')[0] + '.png').convert('RGB'), dtype=np.uint8).copy()
        det_box = None
        with open(file.split('.')[0] + '.txt', 'rb') as f:
            xmint, xmaxt, ymint, ymaxt = [int(float(x)) for x in f.readline().split()]
            det_box = {
                    'x1': xmint - 1280/2,
                    'x2': xmaxt - 1280/2,
                    'y1': 720/2 - ymint ,
                    'y2': 720/2 - ymaxt
            }
        POS_X1  = int(float(det_box['x1'] + det_box['x2'])/2.0)
        POS_Y1  = int(float(det_box['y1'] + det_box['y2'])/2.0)
        WIDTH1  = int(float(det_box['x2'] - det_box['x1']))
        HEIGHT1 = int(float(det_box['y1'] - det_box['y2']))
        # output             = self._detector.detect(frame)
        # POS_X1  = output[0]
        # POS_Y1  = output[1]
        # WIDTH1  = output[2]
        # HEIGHT1 = output[3]

        self.current_frame = 1
        file    = 'data/' + str(self._sequences[self.current_sequence]) + '/' + \
                  str(self.frames_sequences[self.current_sequence][self.current_frame])
        frame   = np.asarray(Image.open(file.split('.')[0] + '.png').convert('RGB'), dtype=np.uint8).copy()
        det_box = None
        with open(file.split('.')[0] + '.txt', 'rb') as f:
            xmint, xmaxt, ymint, ymaxt = [int(float(x)) for x in f.readline().split()]
            det_box = {
                    'x1': xmint - 1280/2,
                    'x2': xmaxt - 1280/2,
                    'y1': 720/2 - ymint ,
                    'y2': 720/2 - ymaxt
            }
        POS_X2  = int(float(det_box['x1'] + det_box['x2'])/2.0)
        POS_Y2  = int(float(det_box['y1'] + det_box['y2'])/2.0)
        WIDTH2  = int(float(det_box['x2'] - det_box['x1']))
        HEIGHT2 = int(float(det_box['y1'] - det_box['y2']))
        # output             = self._detector.detect(frame)
        # POS_X2  = output[0]
        # POS_Y2  = output[1]
        # WIDTH2  = output[2]
        # HEIGHT2 = output[3]

        _state = State()
        _state.DELTA_X = POS_X2 - POS_X1
        _state.DELTA_Y = POS_Y2 - POS_Y1
        print "\n-----Parameters-----"
        print "Delta    X:", _state.DELTA_X
        print "Delta    Y:", _state.DELTA_Y

        self.old_x          = POS_X1
        self.old_y          = POS_Y1
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
                    'x1': xmin - 1280/2,
                    'x2': xmax - 1280/2,
                    'y1': 720/2 - ymin,
                    'y2': 720/2 - ymax
        }

        # TRACKER
        det_box = None
        with open(file, 'rb') as f:
            xmint, xmaxt, ymint, ymaxt = [int(float(x)) for x in f.readline().split()]
            det_box = {
                    'x1': xmint - 1280/2,
                    'x2': xmaxt - 1280/2,
                    'y1': 720/2 - ymint ,
                    'y2': 720/2 - ymaxt
            }
        # output = self._detector.detect(frame)
        # POS_X  = output[0]
        # POS_Y  = output[1]
        # WIDTH  = output[2]
        # HEIGHT = output[3]
        POS_X  = int(float(det_box['x1'] + det_box['x2'])/2.0)
        POS_Y  = int(float(det_box['y1'] + det_box['y2'])/2.0)
        WIDTH  = int(float(det_box['x2'] - det_box['x1']))
        HEIGHT = int(float(det_box['y1'] - det_box['y2']))
        # det_box = {
        #             'x1': POS_X - WIDTH/2,
        #             'x2': POS_X + WIDTH/2,
        #             'y1': POS_Y - HEIGHT/2,
        #             'y2': POS_Y + HEIGHT/2
        # }

        # CUSTOM AGENT
        print "\n-----Parameters-----"
        print "Delta    X:", self.current_state.DELTA_X
        print "Delta    Y:", self.current_state.DELTA_Y
        new_x      = self.old_x + action[0]
        new_y      = self.old_y + action[1]
        self.old_x = POS_X
        self.old_y = POS_Y
        agent_box = {
                    'x1': new_x - WIDTH/2,
                    'x2': new_x + WIDTH/2,
                    'y1': new_y + HEIGHT/2,
                    'y2': new_y - HEIGHT/2
        }

        iou_unconstrained = self.get_iou(gt_box, det_box)
        iou_constrained   = self.get_iou(gt_box, agent_box)
        print "-----IoU Stats-----"
        print "Uncons IoU:", iou_unconstrained
        print "Cons   IoU:", iou_constrained

        dist_x = float(agent_box['x1'] + agent_box['x2'])/2.0 - float(gt_box['x1'] + gt_box['x2'])/2.0
        dist_y = float(agent_box['y1'] + agent_box['y2'])/2.0 - float(gt_box['y1'] + gt_box['y2'])/2.0
        dist = np.linalg.norm([dist_x, dist_y])/self.max_dist
        reward = (1-dist)*self.SCALE_DIST + iou_constrained*self.SCALE_IOU
        print "Distance   :", dist, \
            "\nIoU        :", iou_constrained, \
            "\nDist Reward:", (1-dist)*self.SCALE_DIST, \
            "\nIoU Reward :", iou_constrained*self.SCALE_IOU

        vis_util.draw_bounding_boxes_on_image_array( frame,
                                                     np.array([[(float(ymin)/720.0),
                                                                (float(xmin)/1280.0),
                                                                (float(ymax)/720.0),
                                                                (float(xmax)/1280.0)]]),
                                                     color='black',
                                                     thickness=7)
        vis_util.draw_bounding_boxes_on_image_array( frame,
                                                     np.array([[(float(720/2 - det_box['y1'])/720.0),
                                                                (float(det_box['x1'] + 1280/2)/1280.0),
                                                                (float(720/2 - det_box['y2'])/720.0),
                                                                (float(det_box['x2'] + 1280/2)/1280.0)]]),
                                                     color='blue',
                                                     thickness=3)
        vis_util.draw_bounding_boxes_on_image_array( frame,
                                                     np.array([[(float(720/2 - agent_box['y1'])/720.0),
                                                                (float(agent_box['x1'] + 1280/2)/1280.0),
                                                                (float(720/2 - agent_box['y2'])/720.0),
                                                                (float(agent_box['x2'] + 1280/2)/1280.0)]]),
                                                     color='yellow',
                                                     thickness=5)

        done = 0
        self.current_frame += 1
        if self.current_frame>=self.length_sequences[self.current_sequence]:
            done = 1
            self.current_sequence =  (self.current_sequence + 1)%len(self._sequences)
        print "Action     :", action, "pixels"
        print "Reward     :", reward
        print "Done       :", done

        # imgplot = plt.imshow(frame)
        # plt.show()
        if not self.fig:
            plt.ion()
            self.fig = plt.figure()
            self.plot = plt.subplot(1,1,1)
            plt.imshow(frame)
            ax = self.fig.gca()
            ax.set_xticks(np.arange(0., 1280., 85.33))
            ax.set_yticks(np.arange(0., 720., 48.))
            plt.grid()
            self.fig.show()
        else:
            plt.imshow(frame)
            self.plot.relim()
            self.fig.canvas.flush_events()

        _state = State()
        if not done:
            file   = 'data/' + str(self._sequences[self.current_sequence]) + '/' + \
                     str(self.frames_sequences[self.current_sequence][self.current_frame])
            det_box = None
            with open(file, 'rb') as f:
                xmint, xmaxt, ymint, ymaxt = [int(float(x)) for x in f.readline().split()]
                det_box = {
                        'x1': xmint - 1280/2,
                        'x2': xmaxt - 1280/2,
                        'y1': 720/2 - ymint ,
                        'y2': 720/2 - ymaxt
                }
            # output = self._detector.detect(frame)
            # POS_X  = output[0]
            # POS_Y  = output[1]
            # WIDTH  = output[2]
            # HEIGHT = output[3]
            POS_X  = int(float(det_box['x1'] + det_box['x2'])/2.0)
            POS_Y  = int(float(det_box['y1'] + det_box['y2'])/2.0)
            WIDTH  = int(float(det_box['x2'] - det_box['x1']))
            HEIGHT = int(float(det_box['y1'] - det_box['y2']))

            _state.DELTA_X = POS_X - self.old_x
            _state.DELTA_Y = POS_Y - self.old_y
            self.current_state = _state

        return self.state_to_array(_state), reward, done
