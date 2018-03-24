import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import visualization_utils as vis_util

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('flag', '', 'train or val')
FLAGS = flags.FLAGS


def create_tf_example(image, bbox, im_shape):

    encoded_image_data = None
    with tf.gfile.GFile(image, 'rb') as fid:
        encoded_image_data = fid.read()  # Encoded image bytes

    height       = im_shape[0] # Image height
    width        = im_shape[1] # Image width
    filename     = image # Filename of the image. Empty if image is not from file
    image_format = b'png'

    xmins = [float(bbox.x1)/im_shape[1]] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [float(bbox.x2)/im_shape[1]] # List of normalized right x coordinates in bounding box (1 per box)
    ymins = [float(bbox.y1)/im_shape[0]] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [float(bbox.y2)/im_shape[0]] # List of normalized bottom y coordinates in bounding box (1 per box)

    classes      = [1] # List of integer class id of bounding box (1 per box)
    classes_text = ['car'] # List of string class name of bounding box (1 per box)


    tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        # crop images by -10% to 20% of their height/width
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -50 to +50 percent (per axis)
            rotate=(-45, 45), # rotate by -90 to +90 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
    ],
    random_order=True)

    ann              = None
    num_orig_samples = 25
    num_batches      = 100
    images, bboxs    = [], []

    for i in range(num_orig_samples):
        path = 'data/orig_data/' + FLAGS.flag + '/' + str(i).zfill(4) + '.xml'
        tree = ET.parse(path)
        root = tree.getroot()

        obj = root.findall('object')
        bndbox = obj[0].find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)


        image = np.asarray(Image.open(('data/orig_data/' + FLAGS.flag + '/' + \
                            str(i).zfill(4) + '.png')), dtype='uint8')
        bbox  = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax)], shape=image.shape)

        images.append(image)
        bboxs.append(bbox)
        #image_copy = image.copy()
        #vis_util.draw_bounding_boxes_on_image_array( image_copy,
        #                                                 np.array([[float(ymin)/image.shape[0], float(xmin)/image.shape[1], float(ymax)/image.shape[0], float(xmax)/image.shape[1] ]]),
        #                                                 color='yellow',
        #                                                thickness=4)
       	#imgplot = plt.imshow(image_copy)
        #plt.show()


    for i in range(num_batches):
        seq_det    = seq.to_deterministic()
        aug_images = seq_det.augment_images(images)
        aug_bboxs  = seq_det.augment_bounding_boxes(bboxs)

        for j, (image_np, abbox) in enumerate(zip(aug_images, aug_bboxs)):

        	#aug_bboxs[]
            bbox = abbox.bounding_boxes[0]
            result   = Image.fromarray(image_np)
            out_path = '/home/kshitiz/workspace/autonomous_drone_for_tracking/TF_ObjectDetection/data/' + \
                        FLAGS.flag + '/' + str(i*num_orig_samples+j).zfill(6) + '.png'
            result.save(out_path)

            #image = image_np.copy()
            #vis_util.draw_bounding_boxes_on_image_array( image,
            #                                             np.array([[ float(bbox.y1)/image.shape[0], float(bbox.x1)/image.shape[1], float(bbox.y2)/image.shape[0], float(bbox.x2)/image.shape[1]]]),
            #                                             color='yellow',
            #                                             thickness=4)
            #imgplot = plt.imshow(image)
            #plt.show()


            tf_example = create_tf_example(out_path, bbox, image.shape)
            writer.write(tf_example.SerializeToString())

        if i%10==0 and i!=0:
            print "Number of batches processed", i

    writer.close()


if __name__ == '__main__':
  tf.app.run()
