import os
import sys
import cv2
import tarfile
import zipfile
import numpy as np
import tensorflow as tf
import six.moves.urllib as urllib

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from PIL import Image

python_path = os.path.abspath('TF_ObjectDetection')
sys.path.append(python_path)
python_path = os.path.abspath('TF_ObjectDetection/slim')
sys.path.append(python_path)

from object_detection.utils import ops as utils_ops
if tf.__version__ < '1.5.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.5.* or later!')

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class Detector:
    MODEL_NAME = 'ssd_mobilenet_v1_coco'

    NUM_CLASSES    = 1
    PATH_TO_CKPT   = os.path.join('TF_ObjectDetection', MODEL_NAME, 'frozen_inference_graph.pb')
    PATH_TO_LABELS = os.path.join('TF_ObjectDetection', 'label_map.pbtxt')


    fig              = None
    min_score_thresh = 0.25

    def __init__(self):
        if not os.path.isfile(self.PATH_TO_CKPT):
            raise Exception('Model File not Found')

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        label_map  = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories( label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

    def run_inference_for_single_image(self, image):
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                tensor_dict      = {}
                ops              = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                for key in ['num_detections', 'detection_boxes', 'detection_scores',
                            'detection_classes', 'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection       = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes          = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks          = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks,
                                                                                          detection_boxes,
                                                                                          image.shape[0],
                                                                                          image.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)

                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections']    = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes']   = output_dict['detection_boxes'][0]
                output_dict['detection_scores']  = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def test_detection(self):
        # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
        PATH_TO_TEST_IMAGES_DIR = 'data/detector_val'
        TEST_IMAGE_PATHS        = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '000{}.png'.format(i)) for i in range(0, 3) ]

        for image_path in TEST_IMAGE_PATHS:
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.uint8).copy()

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            # Actual detection.
            output_dict = self.run_inference_for_single_image(image_np)

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array( image_np,
                                                                output_dict['detection_boxes'],
                                                                output_dict['detection_classes'],
                                                                output_dict['detection_scores'],
                                                                self.category_index,
                                                                instance_masks=output_dict.get('detection_masks'),
                                                                use_normalized_coordinates=True,
                                                                line_thickness=2)
            import matplotlib.image as mpimg
            mpimg.imsave(image_path.split('/')[-1], image_np)

    def detect(self, image_np, gt_box=None):
        image = image_np.copy()
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        output_dict = self.run_inference_for_single_image(image_np)

        if gt_box is not None:
            vis_util.draw_bounding_boxes_on_image_array( image,
                                                         np.array([gt_box]),
                                                         color='black',
                                                         thickness=4)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array( image,
                                                            output_dict['detection_boxes'],
                                                            output_dict['detection_classes'],
                                                            output_dict['detection_scores'],
                                                            self.category_index,
                                                            min_score_thresh=self.min_score_thresh,
                                                            instance_masks=output_dict.get('detection_masks'),
                                                            use_normalized_coordinates=True,
                                                            skip_scores=False,
                                                            skip_labels=True,
                                                            line_thickness=4)
        # cv2.imshow('Simulation', image)
        # cv2.waitKey(10)

        bboxes  = output_dict['detection_boxes']
        classes = output_dict['detection_classes']
        scores  = output_dict['detection_scores']

        bboxes   = [bbox for bbox, _ in sorted(zip(bboxes, scores), key=lambda pair: pair[1], reverse=True)]
        classes  = [clss for clss, _ in sorted(zip(classes, scores), key=lambda pair: pair[1], reverse=True)]
        scores   = sorted(scores, key=lambda x: x, reverse=True)

        im_height, im_width = image_np.shape[0:2]
        for i in range(len(bboxes)):
          if scores is None or scores[i] > self.min_score_thresh:
            if classes[i] in self.category_index.keys():
                class_name = self.category_index[classes[i]]['name']
                if class_name=='car':
                    box = tuple(bboxes[i].tolist())
                    ymin, xmin, ymax, xmax = box
                    left   = xmin * im_width
                    right  = xmax * im_width
                    top    = ymin * im_height
                    bottom = ymax * im_height

                    POS_X  = (left + right - im_width)/2.0
                    POS_Y  = (im_height - top - bottom)/2.0
                    WIDTH  = right - left
                    HEIGHT = bottom - top

                    return (POS_X, POS_Y, WIDTH, HEIGHT)

        return None

# if __name__=='__main__':
#     model = Detector()
#     model.test_detection()
