import tensorflow as tf
import numpy as np
import config as cfg

slim = tf.contrib.slim

class DarkNet(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.grid_size = cfg.GRID_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.boxes_per_grid = cfg.BOXES_PER_GRID
        self.output_size = (self.grid_size * self.grid_size) * (self.boxes_per_grid * 5 + self.num_class)
        self.scale = 1.0 * self.image_size / self.grid_size  # normalization
        self.boundary1 = self.grid_size * self.grid_size * self.num_class
        self.boundary2 = self.boundary1 + self.grid_size * self.grid_size * self.boxes_per_grid

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE
        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA

        self.offset = np.transpose(np.reshape(np.array([np.arange(self.grid_size)] * self.grid_size * self.boxes_per_grid),\
                                              (self.boxes_per_grid, self.grid_size, self.grid_size)), (1, 2, 0))
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.logits = self.run_network(self.images, num_outputs=self.output_size, alpha=self.alpha, is_training=is_training)

        if is_training:
            self.labels = tf.placeholder(tf.float32,[None, self.grid_size, self.grid_size, 5 + self.num_class])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)


    def run_network(self, images, num_outputs, alpha, keep_prob=0.5, is_training=True, scope='yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope(
                    [slim.conv2d, slim.fully_connected],
                    activation_fn=leaky_relu(alpha),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            ):
                net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),name='pad_1')
                net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),name='pad_27')
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training,scope='dropout_35')
                net = slim.fully_connected(net, num_outputs, activation_fn=None, scope='fc_36')

            return net


    def calculate_iou(self, predicted_boxes, true_boxes, scope='iou'):
        """

        :param self:
        :param predicted_boxes: 5d tensor[batch_size, grid_size, grid_size, boxes_per_grid, (x, y, w, h)]
        :param true_boxes: 5d tensor[batch_size, grid_size, grid_size, boxes_per_grid, (x, y, w, h)]
        :param scope:
        :return: 4d tensor[batch_size, grid_size, grid_size, boxes_per_grid]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x_upperleft, y_upperleft, x_bottomright, y_bottomright)
            predicted_boxes_t = tf.stack([predicted_boxes[..., 0] - predicted_boxes[..., 2] / 2.0, \
                                          predicted_boxes[..., 1] - predicted_boxes[..., 3] / 2.0, \
                                          predicted_boxes[..., 0] + predicted_boxes[..., 2] / 2.0, \
                                          predicted_boxes[..., 1] + predicted_boxes[..., 3] / 2.0],\
                                          axis=-1)

            true_boxes_t = tf.stack([true_boxes[..., 0] - true_boxes[..., 2] / 2.0, \
                                     true_boxes[..., 1] - true_boxes[..., 3] / 2.0, \
                                     true_boxes[..., 0] + true_boxes[..., 2] / 2.0, \
                                     true_boxes[..., 1] + true_boxes[..., 3] / 2.0],\
                                     axis=-1)

            # get the upperleft and bottomright points of the bounding box
            p_upperleft = tf.maximum(predicted_boxes_t[..., :2], true_boxes_t[..., :2])
            p_bottomright = tf.maximum(predicted_boxes_t[..., 2:], true_boxes_t[..., 2:])

            # intersection
            intersection = tf.maximum(0.0, p_bottomright - p_upperleft)
            inter_box = intersection[..., 0] * intersection[..., 1]

            # union
            predicted_box = predicted_boxes[..., 2] * predicted_boxes[..., 3]
            true_box = true_boxes[..., 2] * true_boxes[..., 3]
            union_box = tf.maximum(predicted_box + true_box - inter_box, 1e-10)

        return tf.clip_by_value(inter_box / union_box, 0.0, 1.0)


    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(predicts[:, :self.boundary1],\
                                         [self.batch_size, self.grid_size, self.grid_size, self.num_class])
            predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2],\
                                        [self.batch_size, self.grid_size, self.grid_size, self.boxes_per_grid])
            predict_boxes = tf.reshape(predicts[:, self.boundary2:],\
                                       [self.batch_size, self.grid_size, self.grid_size, self.boxes_per_grid, 4])

            # reshape labels to match the tensor size of predicts
            response = tf.reshape(labels[..., 0],[self.batch_size, self.grid_size, self.grid_size, 1])
            boxes = tf.reshape(labels[..., 1:5],[self.batch_size, self.grid_size, self.grid_size, 1, 4])
            boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_grid, 1]) / self.image_size
            classes = labels[..., 5:]

            # adjust boxes with offset
            offset = tf.reshape(tf.constant(self.offset, dtype=tf.float32),\
                                [1, self.grid_size, self.grid_size, self.boxes_per_grid])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))
            predict_boxes_train = tf.stack([(predict_boxes[..., 0] + offset) / self.grid_size,\
                                           (predict_boxes[..., 1] + offset_tran) / self.grid_size,\
                                           tf.square(predict_boxes[..., 2]),\
                                           tf.square(predict_boxes[..., 3])], axis=-1)

            iou_predict_truth = self.calculate_iou(predict_boxes_train, boxes)

            # calculate I tensor [BATCH_SIZE, GRID_SIZE, GRID_SIZE, BOXES_PER_GRID]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keepdims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [GRID_SIZE, GRID_SIZE, BOXES_PER_GRID]
            noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack([boxes[..., 0] * self.grid_size - offset,\
                                   boxes[..., 1] * self.grid_size - offset_tran,\
                                   tf.sqrt(boxes[..., 2]), tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),\
                                        name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),\
                                         name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),\
                                           name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),\
                                        name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)




def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op














