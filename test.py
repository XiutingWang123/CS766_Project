import os
import csv
import cv2
import argparse
import numpy as np
import tensorflow as tf
import config as cfg
from darknet import DarkNet
from utils.timer import Timer
from utils.process_pascal_voc import pascal_voc
from utils.timer import Timer


class Evaluater(object):

    def __init__(self, network, weight_file, data):
        self.network = network
        self.weight_file = weight_file
        self.data = data

        #config
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
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
        self.saver = tf.train.import_meta_graph(self.weight_file + '/YOLO_train.ckpt-15000.meta')
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.weight_file))



    def interpret_output(self, output):
        """
        Process network output, transforming to bounding box and class probability
        :param output: a np array with shape (1, (self.grid_size * self.grid_size * (self.boxes_per_grid * 5 + self.num_class))
        :return: a list of list info that contains bounding box [[prob, x, y, w, h, class(int)],...]
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
            res.append([probs_filtered[i],boxes_filtered[i][0],\
                        boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3], classes_num_filtered[i]])

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


    def compute_mAP(self, predicted_res, gt_res):
        """

        :param predicted_res: dict {img_id: [prob, x, y, w, h, class(int)]}
        :param gt_res: dict {img_id: [prob, x, y, w, h, class(int)]}
        :return: mean Average precision, average precision
        """
        dictPredicted = {}

        for i in range(self.num_class):
            dictPredicted[i] = []

        totalPredicted = np.zeros(self.num_class, dtype=int)
        totalGT = np.zeros(self.num_class, dtype=int)

        # process predicted_res
        # each index of values in key (class):value pair would consist of [prob, img_id, [x,y,w,h]]
        for img_id, values in predicted_res.items():
            for item in values:
                classId = item[5]
                dictPredicted[classId].append([item[0], img_id, item[1:5]])
                totalPredicted[classId] += 1

        # for each predicted column, sort according to confidence
        for i in range(self.num_class):
            dictPredicted[i].sort(key=lambda x: x[0], reverse=True)

        # process gt_res
        # dict of dict, key: class, nested key : img_id, value: [[x,y,w,h]...]
        dictGT = {}
        dictMask = {}
        for i in range(self.num_class):
            dictGT[i] = {}
            dictMask[i] = {}

        for img_id, values in gt_res.items():
            for item in values:
                classId = item[5]
                if img_id not in dictGT[classId]:
                    dictGT[classId][img_id] = []
                    dictMask[classId][img_id] = []

                dictGT[classId][img_id].append(item[1:5])
                dictMask[classId][img_id].append(0)
                totalGT[classId] += 1

        truePos = []
        falsePos = []
        falseNeg = np.zeros(self.num_class, dtype=int)

        for classId in range(self.num_class):
            numOfPredictedObjInClass = totalPredicted[classId]
            truePos.append(np.zeros(numOfPredictedObjInClass, dtype=int))
            falsePos.append(np.zeros(numOfPredictedObjInClass, dtype=int))

            for i in range(numOfPredictedObjInClass):
                predicted_item = dictPredicted[classId][i]
                # if no object in the corresponding image in ground truth
                if predicted_item[1] not in dictGT[classId]:
                    falsePos[classId][i] = 1
                    continue
                # find the ground truth bounding box corresponding with thepredicted bounding box
                maxIOU = 0.
                maxIndex = -1
                for j in range(len(dictGT[classId][predicted_item[1]])):
                    if dictMask[classId][predicted_item[1]][j] == 1:
                        continue

                    iou = self.compute_iou(predicted_item[2], dictGT[classId][predicted_item[1]][j])

                    if iou > maxIOU:
                        maxIOU = iou
                        maxIndex = j

                if maxIndex == -1:
                    falsePos[classId][i] = 1
                    continue
                if maxIOU > self.iou_threshold:
                    dictMask[classId][predicted_item[1]][maxIndex] = 1
                    truePos[classId][i] = 1
                else:
                    falseNeg[classId] += 1


        # compute average precision for each class
        cumulativePrecision = []
        cumulativeRecall = []
        averagePrecision = np.zeros(self.num_class)
        for classId in range(self.num_class):
            # Cumulative precision : precision with increasing number of detections considered
            #print("length of truePos[{0}]: {1}, length of totalPredicted[{2}]: {3}".format(classId,truePos[classId].shape, classId, totalPredicted[classId].shape))
            cumulativePrecision.append(np.divide(np.cumsum(truePos[classId]), 1 + np.arange(totalPredicted[classId])))
            # Cumulative Recall : recall with increasing number of detections considered
            cumulativeRecall.append(np.cumsum(truePos[classId]) / totalGT[classId])

            recallValues = np.unique(cumulativeRecall[-1])

            for idx, recallThreshold in enumerate(recallValues):
                # Interpolated area under curve for recall value
                if idx == 0:
                    recallStep = recallValues[0]
                else:
                    recallStep = recallValues[idx] - recallValues[idx-1]
                averagePrecision[classId] \
                    += np.max(cumulativePrecision[-1][cumulativeRecall[-1] >= recallThreshold]) * recallStep

        meanAveragePrecision = np.mean(averagePrecision)

        print("Mean Average Precision : %0.4f" % meanAveragePrecision)
        print("{0:>12}".format("ClassName"),
                "{0:7}".format("Ground Truth"),
                "{0:9}".format("Predicted"),
                "{0:13}".format("TruePositives"),
                "{0:13}".format("FalsePositives"),
                "{0:13}".format("FalseNegatives"),
                "{0:12}".format("AvgPrecision"))
        for classId in range(self.num_class):
            print("{0:>12}".format(self.classes[classId]),
                    "{0:>7}".format(totalGT[classId]),
                    "{0:>9}".format(len(dictPredicted[classId])),
                    "{0:>13}".format(np.sum(truePos[classId])),
                    "{0:>13}".format(np.sum(falsePos[classId])),
                    "{0:>13}".format(falseNeg[classId]),
                    "{0:>12.4f}".format(averagePrecision[classId]))


        path_r = os.path.join(self.cache_path, 'cumulativeRecall.csv')
        path_p = os.path.join(self.cache_path, 'cumulativePrecision.csv')
        self.write_list_to_csv(path_r, cumulativeRecall)
        self.write_list_to_csv(path_p, cumulativePrecision)


        """
        if plotPRCurve:
            for classId, className in enumerate(self.classes):
                plt.plot(cumulativeRecall[classId], cumulativePrecision[classId], label=className, c=np.random.rand(3, 1))
            plt.xlim([0, 1])
            plt.ylim([0.5, 1])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc='right', fontsize=11)
            plt.show()
        """

        return meanAveragePrecision, averagePrecision


    def write_list_to_csv(self,filename, my_list):
        with open(filename, 'w') as f:
            wr = csv.writer(f)
            wr.writerow(my_list)



    def write_dict_to_csv(self, filename, my_dict):
        with open(filename, 'w') as f:
            wr = csv.DictWriter(f, my_dict.keys())
            wr.writeheader()
            wr.writerow(my_dict)


    def test(self):

        data_size = len(self.data.gt_labels)

        img_num = 0
        predicted_res = {}
        gt_res = {}
        batch = 1
        test_timer = Timer()

        while img_num < data_size:
            print("Load test images batch %d" % batch)

            images, labels = self.data.get_data()  #labels shape = (64,7,7,25)

            test_timer.start_timer()
            net_output = self.sess.run(self.network.logits, feed_dict={self.network.images: images})
            test_timer.end_timer()

            #print("shape of net_output: {}".format(net_output.shape))
            print("Speed: {}".format(test_timer.average_time))

            for i in range(net_output.shape[0]):
                predicted_res[img_num] = self.interpret_output(net_output[i])
                #gt_res[img_num] = [labels[i, x, y, :] for x in range(self.grid_size) for y in range(self.grid_size) if labels[i, x, y, 0] == 1]

                gt_res[img_num] = []
                # process gt_labels, label = [prob, x, y, w, h, class(20)] to [prob, x, y, w, h, class(int)]
                for x in range(self.grid_size):
                    for y in range(self.grid_size):
                        if labels[i, x, y, 0] == 1:
                            classes = labels[i, x, y, 5:]
                            class_ind = [c for c in range(len(classes)) if classes[c] == 1]
                            gt_res[img_num].append([labels[i, x, y, 0], labels[i, x, y, 1], labels[i, x, y, 2], labels[i, x, y, 3], \
                                                    labels[i, x, y, 4], class_ind[0]])

                #print("length of predicted_res {0}: {1}".format(img_num, len(predicted_res[img_num])))
                #print("length of predicted_res value: {}".format(len(predicted_res[img_num][0])))
                #print("length of gt_res {0}: {1}".format(img_num, len(gt_res[img_num])))
                #print("length of gt_res value: {}".format(len(gt_res[img_num][0])))

                img_num += 1
            batch += 1
        """    
        path_predicted = os.path.join(self.cache_path, 'predicted_res.csv')
        path_gt = os.path.join(self.cache_path, 'gt_res.csv')
        self.write_dict_to_csv(path_predicted, predicted_res)
        self.write_dict_to_csv(path_gt, gt_res)
        """
        self.compute_mAP(predicted_res, gt_res)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='', type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    if args.gpu is not '':
        cfg.GPU = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    darknet = DarkNet(False)
    weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    data = pascal_voc('test')
    evaluater = Evaluater(darknet, weight_file, data)


    print('==== Start evaluation ====')
    evaluater.test()
    print('==== Finish evaluation ====')

    """
    predicted_res = {0:[[0.5276423692703247, 212.23724, 209.12057, 414.2071, 393.7453, 0]], 1:[[0.42673420906066895, 279.03107, 275.76355, 260.0471, 317.2824, 1]]}
    gt_res = {0:[[1.0, 197.568, 191.14666666666668, 395.136, 358.40000000000003, 0]], 1:[[1.0, 259.84000000000003, 254.46400000000003, 281.344, 367.95733333333334, 1]]}
    evaluater.compute_mAP(predicted_res, gt_res)
    """





if __name__ == '__main__':
    # argument: python test.py --weights YOLO_small.ckpt --gpu 0
    # argument: python test.py
    main()
