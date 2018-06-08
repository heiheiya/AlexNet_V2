import numpy as np
import cv2

class ImageDataGenerator:
    def __init__(self, class_list, horizontal_filp=False, shuffle=False, mean=np.array([104., 117., 124.]), scale_size=(227, 227), num_classes=2):
        self.HORIZONTAL_FILP = horizontal_filp
        self.NUM_CLASSES = num_classes
        self.SHUFFLE = shuffle
        self.MEAN = mean
        self.SCALE_SIZE = scale_size
        self.POINTER = 0

        self.read_class_list(class_list)

        if self.SHUFFLE:
            self.shuffle_data()

    def read_class_list(self, class_list):
        with open(class_list) as f:
            lines = f.readlines()
            self.images = []
            self.labels = []
            for l in lines:
                items = l.split()
                self.images.append(items[0])
                self.labels.append(int(items[1]))

            self.data_size = len(self.labels)

    def shuffle_data(self):
        images = self.images.copy()
        labels = self.labels.copy()
        self.images = []
        self.labels = []

        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def reset_pointer(self):
        self.POINTER = 0
        if self.SHUFFLE:
            self.shuffle_data()

    def next_batch(self, batch_size):
        paths = self.images[self.POINTER:self.POINTER+batch_size]
        labels = self.labels[self.POINTER:self.POINTER+batch_size]

        self.POINTER += batch_size

        images = np.ndarray([batch_size, self.SCALE_SIZE[0], self.SCALE_SIZE[1], 3])
        for i in range(len(paths)):
            img = cv2.imread(paths[i])
            if self.HORIZONTAL_FILP and np.random.random() < 0.5:
                img = cv2.flip(img, 1)

            img = cv2.resize(img, (self.SCALE_SIZE[0], self.SCALE_SIZE[1]))
            img = img.astype(np.float32)

            img -= self.MEAN
            images[i] = img

        one_hot_labels = np.zeros((batch_size, self.NUM_CLASSES))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1

        return images, one_hot_labels