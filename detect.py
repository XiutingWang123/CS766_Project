import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import config as cfg
from darknet import DarkNet
from utils.timer import Timer


class Detector(object):

    def __init__(self, network, weight_file):
        self.network = network
        self.weight_file = weight_file
        self.outputFile = cfg.PASCAL_PATH_TEST

        #config
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.grid_size = cfg.GRID_SIZE
        self.boxes_per_grid = cfg.BOXES_PER_GRID
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.iou_overlap_threshold = cfg.IOU_OVERLAP_THRESHOLD

        # boundaries for separating logits
        self.boundary1 = self.grid_size * self.grid_size * self.num_class
        self.boundary2 = self.boundary1 + self.grid_size * self.grid_size * self.boxes_per_grid

        # run sess
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: {}'.format(self.weight_file))
        self.saver = tf.train.Saver()
        #self.saver.restore(self.sess, self.weight_file)
        self.saver = tf.train.import_meta_graph(self.weight_file + 'YOLO_train.ckpt-15000.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.weight_file))

    def image_detector(self, imname):

        image = cv2.imread(imname) #BGR
        res = self.detect(image)
        self.draw_boxes(image, res)
        cv2.imshow('Image', image)
        cv2.waitKey(0)


    def detect(self, image):
        img_h, img_w, _ = image.shape
        inputs = self.process(image)
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))
        net_output = self.sess.run(self.network.logits, feed_dict={self.network.images: inputs})
        res = []

        #print("shape of net_output: {}".format(net_output.shape))
        for i in range(net_output.shape[0]):
            res.append(self.interpret_output(net_output[i]))

        res = res[0]
        for i in range(len(res)):
            res[i][1] *= (1.0 * img_w / self.image_size) #x
            res[i][2] *= (1.0 * img_h / self.image_size) #y
            res[i][3] *= (1.0 * img_w / self.image_size) #w
            res[i][4] *= (1.0 * img_h / self.image_size) #h

        return res


    def process(self, image):
        """
        Process image [B, G, R] to [R, G, B] and rescale to -1 ~ 1
        :param image: image [B, G, R]
        :return: image with (self.image_size * self.image_size) and [R, G, B] order
        """
        inputs = cv2.resize(image, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32) # BGR to RGB
        inputs = (inputs / 255.0) * 2.0 - 1.0  # scale to -1 ~ 1

        return inputs


    def interpret_output(self, output):
        """
        Process network output, transforming to bounding box and class probability
        :param output: a np array with shape [1, (self.grid_size * self.grid_size * (self.boxes_per_grid * 5 + self.num_class)]
        :return: a list of list info that contains bounding box [[class(str), x, y, w, h, prob],...]
        """

        probs = np.zeros((self.grid_size, self.grid_size, self.boxes_per_grid, self.num_class))
        class_probs = np.reshape(output[0:self.boundary1], (self.grid_size, self.grid_size, self.num_class))
        scales = np.reshape(output[self.boundary1:self.boundary2],(self.grid_size, self.grid_size, self.boxes_per_grid))
        boxes = np.reshape(output[self.boundary2:], (self.grid_size, self.grid_size, self.boxes_per_grid, 4)) # 4 = [x,y,w,h]
        offset = np.transpose(np.reshape(np.array([np.arange(self.grid_size)] * self.grid_size * self.boxes_per_grid), \
                                         (self.boxes_per_grid, self.grid_size, self.grid_size)), (1, 2, 0))
        # adjust boxes with offset
        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.grid_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        # scale up to match image size
        boxes *= self.image_size

        # compute the probability
        for i in range(self.boxes_per_grid):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        probs_isGreater = np.array(probs >= self.threshold, dtype='bool')
        boxes_matchIdx = np.nonzero(probs_isGreater)
        boxes_filtered = boxes[boxes_matchIdx[0], boxes_matchIdx[1], boxes_matchIdx[2]]
        probs_filtered = probs[probs_isGreater]

        classes_num_filtered = np.argmax(probs_isGreater, axis=3)[boxes_matchIdx[0], boxes_matchIdx[1], boxes_matchIdx[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]  # decreasing order
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]


        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                # delete overlap bounding boxes
                if self.compute_iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_overlap_threshold:
                    probs_filtered[j] = 0.0

        iou_isGreater = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[iou_isGreater]
        probs_filtered = probs_filtered[iou_isGreater]
        classes_num_filtered = classes_num_filtered[iou_isGreater]

        res = []
        for i in range(len(boxes_filtered)):
            res.append([self.classes[classes_num_filtered[i]],boxes_filtered[i][0],\
                        boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

        return res


    def compute_iou(self, box1, box2):
        """

        :param box1: [x,y,w,h]
        :param box2: [x,y,w,h]
        :return: float, IOU
        """
        inter_w = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        inter_h = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if inter_w <= 0 or inter_h <= 0 else inter_w * inter_h

        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)


    def draw_boxes(self, image, res):

        for i in range(len(res)):
            x = int(res[i][1])
            y = int(res[i][2])
            w = int(res[i][3] * 0.5)
            h = int(res[i][4] * 0.5)
            # unique color for each class
            seed = [350, 750, 180]
            color = tuple([(j * (1 + self.classes.index(res[i][0])) % 255) for j in seed])
            cv2.rectangle(image, (x - w, y - h), (x + w, y + h), color, 2)
            cv2.rectangle(image, (x - w, y - h), (x + w, y - h + 20), color, -1)
            lineType = cv2.LINE_AA
            cv2.putText(image, res[i][0] + ' : %.2f' % res[i][5],(x - w + 5, y - h + 15), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType)

    """
    def video_detector(self, cap):

        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            result = self.detect(frame)
            self.draw_boxes(frame, result)
            # Display the resulting frame
            cv2.imshow('Video', frame)
            cv2.waitKey(10)
            ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()
    """

    def video_detector(self, cap):

        # get the information of cap
        codec = cv2.VideoWriter_fourcc(*'MJPG') #int(cap.get(cv2.CAP_PROP_FOURCC))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # output stream
        output = cv2.VideoWriter(self.outputFile + '/video_output.avi', codec, fps, (frameWidth,frameHeight))
        frameIndex = cap.get(cv2.CAP_PROP_POS_FRAMES)
        totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        timer = Timer()

        while True:
            ret, frame = cap.read()
            if ret:
                timer.start_timer()
                result = self.detect(frame)
                timer.end_timer()
                frameIndex = cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.draw_boxes(frame, result)
                output.write(frame)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex-1)
                cv2.waitKey(100)

            if frameIndex == totalFrames:
                break

        cap.release()
        output.release()
        cv2.destroyAllWindows()
        print("Average time for object detection per frame: {:.3f} fps".format(1/timer.average_time))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='', type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--test_dir', default='test', type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--image', default='', type=str, help='path to test image')
    parser.add_argument('--video',  default='', type=str, help='path to test video')
    args = parser.parse_args()

    if args.gpu is not '':
        cfg.GPU = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    net = DarkNet(False)
    weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    test_path = os.path.join(args.data_dir, args.test_dir)
    detector = Detector(net, weight_file)

    if args.image is not '':
        imname = os.path.join(test_path, args.image)
        detector.image_detector(imname)

    if args.video is not '':
        video = os.path.join(test_path, args.video)
        cap = cv2.VideoCapture(video)
        detector.video_detector(cap)



if __name__ == '__main__':
    # argument: python detect.py --weights YOLO_small.ckpt --image dog.jpg
    # argument: python detect.py  --image dog.jpg
    # argument: python detect.py  --video traffic.avi
    main()
