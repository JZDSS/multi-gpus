import tensorflow as tf


def smooth_l1_loss(x):
    """
    Computes smooth l1 loss: x^2 / 2 if abs(x) < 1, abs(x) - 0.5 otherwise.
    See [Fast R-CNN](https://arxiv.org/abs/1504.08083)
    :param x:
    :return:
    """
    square_loss = 0.5 * x ** 2
    absolute_loss = tf.abs(x)
    loss = tf.where(tf.less(absolute_loss, 1.), square_loss, absolute_loss - 0.5)
    tf.add_to_collection(tf.GraphKeys.LOSSES, loss)


def random_flip_left_right_with_bbox():
    pass


def _transform_bboxes(bboxes, bbox_ref):
    bbox_ref = tf.reshape(bbox_ref, [-1])
    v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
    transformed_bboxes = bboxes - v
    s = tf.stack([bbox_ref[2] - bbox_ref[0],
                  bbox_ref[3] - bbox_ref[1],
                  bbox_ref[2] - bbox_ref[0],
                  bbox_ref[3] - bbox_ref[1]])
    transformed_bboxes = transformed_bboxes / s
    transformed_bboxes = tf.nn.relu(transformed_bboxes)
    transformed_bboxes = 1 - tf.nn.relu(1 - transformed_bboxes)
    return transformed_bboxes


def random_crop_with_bbox(img, bboxes, minimum_jaccard_overlap=0.7, aspect_ratio_range=(0.5, 2), area_range=(0.1, 1.0)):
    """
    Random crop the image and transfrom the bounding boxes to associated with the cropped image.

    :param img: The original image, a 3-D Tensor with shape [height, width, channels].
    :param bboxes: 2-D Tensor of type tf.float32 with shape [num_boxes, 4], the coordinates  of 2ed dimension
        is ordered [min_y, min_x, max_y, max_x], which are normalized to [0, 1] by dividing image height and width.
    :param minimum_jaccard_overlap: A Tensor of type float32. Defaults to 0.7. The cropped area of the image must
        contain at least this fraction of any bounding box supplied. The value of this parameter should be non-negative.
        In the case of 0, the cropped area does not need to overlap any of the bounding boxes supplied.
    :param aspect_ratio_range: An optional list of floats. Defaults to [0.5, 2]. The cropped area of the image
        must have an aspect ratio = width / height within this range.
    :param area_range: An optional list of floats. Defaults to [0.1, 1]. The cropped area of the image must contain
        a fraction of the supplied image within in this range.
    :return: Cropped image and transformed bounding boxes.
    """
    begin, size, bbox_for_slice = tf.image.sample_distorted_bounding_box(
                                tf.shape(img),
                                bounding_boxes=tf.expand_dims(bboxes, 0),
                                min_object_covered=minimum_jaccard_overlap,
                                aspect_ratio_range=aspect_ratio_range,
                                area_range=area_range,
                                use_image_if_no_bounding_boxes=True)
    cropped_image = tf.slice(img, begin, size)
    transformed_bboxes = _transform_bboxes(bboxes, bbox_for_slice)
    return cropped_image, transformed_bboxes


def main():
    import cv2
    import numpy as np
    img = cv2.imread('/home/yqi/Desktop/workspace/PycharmProjects/VOCdevkit/VOC2007/JPEGImages/000005.jpg')
    bbx = [[0.562, 0.526, 0.904, 0.648],
           [0.650, 0.01, 0.997, 0.134]]
    x = tf.placeholder(tf.float32, shape=[None, None, 3])
    b = tf.placeholder(tf.float32, shape=[2, 4])
    cropped, transformed = random_crop_with_bbox(x, b)
    drawed = tf.image.draw_bounding_boxes(tf.expand_dims(cropped, 0), tf.expand_dims(transformed, 0))
    with tf.Session() as sess:
        while True:
            t, to_show = sess.run([transformed, drawed], feed_dict={x: img, b: bbx})
            print(t)
            cv2.imshow("a", to_show.astype(np.uint8)[0])
            cv2.waitKey(0)

if __name__ == '__main__':
    main()