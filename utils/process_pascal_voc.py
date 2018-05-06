import cv2
import os
import copy
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import config as cfg


class pascal_voc(object):
    def __init__(self, phase):
        self.phase = phase
        # initialize data path
        if self.phase == 'train':
            self.flipped = True
            self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
        else:
            self.flipped = False
            self.devkil_path = os.path.join(cfg.PASCAL_PATH_TEST, 'VOCdevkit')

        self.data_path = os.path.join(self.devkil_path, 'VOC2007')
        self.cache_path = cfg.CACHE_PATH

        # config
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.grid_size = cfg.GRID_SIZE
        self.classes = cfg.CLASSES
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None # ground truth labels
        self.process_data()


    def process_data(self):
        gt_labels = self.load_labels()

        # if flipped (is true), append horizontally-flipped training examples
        if self.flipped:
            print('Appending horizontally-flipped training examples.')
            gt_labels_copy = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_copy)):
                gt_labels_copy[idx]['flipped'] = True
                gt_labels_copy[idx]['label'] = gt_labels_copy[idx]['label'][:, ::-1, :]
                # update x value
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        if gt_labels_copy[idx]['label'][i, j, 0] == 1:
                            gt_labels_copy[idx]['label'][i, j, 1] = self.image_size - 1 - gt_labels_copy[idx]['label'][i,j,1]
            gt_labels += gt_labels_copy
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels


    def load_labels(self):
        """

        :return: gt_labels, a list of dict, ordering based on image index
        """
        cache_file = os.path.join(self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')

        if os.path.isfile(cache_file):
            print('Loading gt_labels from: {}'.format(cache_file))
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels


        print('Processing gt_lables from: {}'.format(self.data_path))

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)


        if self.phase == 'train':
            filename = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            filename = os.path.join(self.data_path, 'ImageSets', 'Main', 'test.txt')

        with open(filename, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            if num == 0: # no objects
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({'imname': imname, 'label': label, 'flipped': False})

        # write gt_labels to cache_file
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)

        return gt_labels


    def load_pascal_annotation(self, index):
        """
        Extract bounding box information from XML file in the PASCAL VOC format
        :param index: an integer, index of image
        :return: a np array: label [grid_y, grid_x,[prob, x, y, w, h, class(20)]], an integer: number of objects in a given image
        """
        label = np.zeros((self.grid_size, self.grid_size, 25))
        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]

        for obj in objs:
            bbox = obj.find('bndbox')
            # adjust bounding box pixel position to 0-based
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            class_idx = self.class_to_ind[obj.find('name').text.lower().strip()]
            box = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1] #[x,y,w,h]
            x_idx = int(box[0] * self.grid_size / self.image_size)
            y_idx = int(box[1] * self.grid_size / self.image_size)

            if label[y_idx, x_idx, 0] == 1:
                continue
            label[y_idx, x_idx, 0] = 1
            label[y_idx, x_idx, 1:5] = box
            label[y_idx, x_idx, 5+class_idx] = 1

        return label, len(objs)


    def get_data(self):
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.batch_size, self.grid_size, self.grid_size, 25))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.get_image(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels



    def get_image(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0  # scale to -1 ~ 1

        if flipped:
            image = image[:, ::-1, :] # horizontally flipped image

        return image






