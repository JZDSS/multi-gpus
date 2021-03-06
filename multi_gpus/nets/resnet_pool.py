from multi_gpus import layers2 as layers
import tensorflow as tf


def block(inputs, num_outputs, weight_decay, scope, is_training, down_sample = False):
    with tf.variable_scope(scope):
        num_inputs = inputs.get_shape().as_list()[3]

        res = layers.conv(inputs, num_outputs = num_outputs, kernel_size=[3, 3],
                          strides=[1, 2, 2, 1] if down_sample else [1, 1, 1, 1],
                          scope='conv1', b_norm=True, is_training=is_training, weight_decay=weight_decay)

        res = layers.conv(res, num_outputs=num_outputs, kernel_size=[3, 3], activation_fn=None,
                          scope='conv2', b_norm=True, is_training=is_training, weight_decay=weight_decay)
        if  num_inputs != num_outputs:
            inputs = layers.conv(inputs, num_outputs=num_outputs, kernel_size=[1, 1], activation_fn=None,
                                 scope='short_cut', strides=[1, 2, 2,1 ], b_norm=True, is_training=is_training,
                                 weight_decay=weight_decay)
        res = tf.nn.relu(res + inputs)

    return res


def build_net(x, is_training, FLAGS):
    n = FLAGS.blocks
    # shape = x.get_shape().as_list()
    with tf.variable_scope('pre'):
        pre = layers.conv(x, num_outputs=32,  kernel_size = [3, 3], scope='conv', b_norm=True, is_training=is_training,
                          weight_decay=FLAGS.weight_decay)
        # pre = layers.max_pool2d(pre, [2, 2], padding='SAME', scope='pool')
    h = pre
    # for i in range(1, n + 1):
    h = block(h, 32, FLAGS.weight_decay, '32_block{}'.format(1), is_training)

    h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')
    # h = block(h, 64, FLAGS.weight_decay, '64_block1', is_training, True)
    h = layers.conv(h, num_outputs = 64, kernel_size=[3, 3], strides=[1, 1, 1, 1],
                    scope='conv1', b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay)
    
    # for i in range(2, n + 1):
    h = block(h, 64, FLAGS.weight_decay, '64_block{}'.format(1), is_training)

    h = tf.nn.max_pool(h, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')
    h = layers.conv(h, num_outputs = 128, kernel_size=[3, 3], strides=[1, 1, 1, 1],
                    scope='conv2', b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay)
    
    # for i in range(2, n + 1):
    h = block(h, 128, FLAGS.weight_decay, '128_block{}'.format(1), is_training)

    shape = h.get_shape().as_list()

    h = tf.contrib.layers.avg_pool2d(h, [shape[1], shape[2]], scope='global_pool')
    if FLAGS.p_relu:
        h = layers.p_relu(h)
    h = layers.conv(h, num_outputs=FLAGS.num_classes, kernel_size=[1, 1], scope='fc1', padding='VALID',
                    b_norm=True, is_training=is_training, weight_decay=FLAGS.weight_decay, activation_fn=None)

    return tf.reshape(h, [-1, FLAGS.num_classes])


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_integer('num_classes', 10, '')
    flags.DEFINE_float('weight_decay', 0.00004, '')
    FLAGS = flags.FLAGS
    x = tf.placeholder(tf.float32, (10, 10, 10, 3))
    build_net(x, 1, True, FLAGS)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('logs', sess.graph)
        writer.flush()

    writer.close()

