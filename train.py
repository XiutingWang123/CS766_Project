import os
import argparse
import datetime
import tensorflow as tf
import config as cfg
from darknet import DarkNet
from utils.timer import Timer
from utils.process_pascal_voc import pascal_voc

slim = tf.contrib.slim


class Solver(object):

    def __init__(self, network, data):
        self.network = network
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER


        # initialize learning rate parameters
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE

        # initialize saving directory
        self.output_dir = os.path.join(cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y:%m:%d:%H:%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_cfg()
        self.ckpt_file = os.path.join(self.output_dir, 'YOLO_train.ckpt')

        # set up summary and writer
        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        # set up training parameters
        self.global_step = tf.train.create_global_step() # number of batches seen by the graph
        self.learning_rate = tf.train.exponential_decay(self.initial_learning_rate,\
                                                        self.global_step,self.decay_steps, self.decay_rate,\
                                                        self.staircase, name='learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = slim.learning.create_train_op(self.network.total_loss, self.optimizer, global_step=self.global_step)
        config = tf.ConfigProto(gpu_options=tf.GPUOptions())
        self.sess = tf.Session(config=config)

        # initialize variables of network
        self.sess.run(tf.global_variables_initializer())

        if self.weights_file is not None:
            print('Restoring weights from {}'.format(self.weights_file))
            #self.saver.restore(self.sess, self.weights_file)
            self.saver = tf.train.import_meta_graph(self.weights_file + '/YOLO_train.ckpt-15000.meta')
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.weights_file))

        self.writer.add_graph(self.sess.graph)



    def train(self):

        timer = Timer()
        for iter in range(1, self.max_iter + 1):
            images, labels = self.data.get_data()
            feed_dict = {self.network.images: images, self.network.labels: labels}

            if iter % self.summary_iter == 0:
                timer.start_timer()
                summary, loss, _ = self.sess.run([self.summary_op, self.network.total_loss, self.train_op], feed_dict=feed_dict)
                timer.end_timer()

                print('Epoch: {}, Iter: {}, Learning rate: {}, Loss: {}, Speed: {}, Remain: {}'.\
                      format(self.data.epoch, iter, round(self.learning_rate.eval(session=self.sess), 6),\
                             loss, timer.average_time, timer.remain(iter, self.max_iter)))

                self.writer.add_summary(summary, iter)

            else:
                self.sess.run(self.train_op, feed_dict=feed_dict)


            if iter % self.save_iter == 0:
                print('iter {}: save checkpoint file to {}'.format(iter, self.ckpt_file))
                self.saver.save(self.sess, self.ckpt_file, global_step=self.global_step)



    def output_cfg(self):
        cfg_dict = cfg.__dict__
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            for key, value in cfg_dict.items():
                if key.isupper():
                    info = '{}: {}\n'.format(key, value)
                    f.write(info)



def update_cfgpath(data_dir, weights_file):
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')
    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='', type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    if args.data_dir != cfg.DATA_PATH:
        update_cfgpath(args.data_dir, args.weights)

    if args.gpu is not '':
        cfg.GPU = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    solver = Solver(DarkNet(), pascal_voc('train'))

    print('==== Start Training ====')
    solver.train()
    print('==== Finish Training ====')

if __name__ == '__main__':
    # argument: python train.py --weights YOLO_small.ckpt --gpu 0
    # argument: python train.py --gpu 0
    main()


