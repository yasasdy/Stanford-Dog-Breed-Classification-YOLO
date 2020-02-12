import tensorflow as tf
import numpy as np
import os
import cv2
import colorsys
import argparse
import yolo.config as cfg
from yolo.yolo_vgg16 import YOLONet
import matplotlib.pyplot as plt
import glob
from utils.preprocess import preprocess
import xml.etree.ElementTree as ET


class Detector(object):
    def _init_(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    def draw_result(self, img, result):
        colors = self.random_colors(len(result))
        name=''
        x=0
        y=0
        w=0
        h=0
        con=0
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            color = tuple([rgb * 255 for rgb in colors[i]])
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), color, 1)
            name=result[i][0]
            cv2.putText(img, result[i][0], (x - w + 1, y - h + 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
            print(result[i][0],': %.2f%%' % (result[i][5]*100))
            con=result[i][5]
        return (name,con,x,y,w,h)

    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits, feed_dict={self.net.images: inputs})
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
        probs = np.zeros((self.cell_size, self.cell_size, self.boxes_per_cell, self.num_class))
        class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))
        scales = np.reshape(output[self.boundary1:self.boundary2], (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                         [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))#7*7*2

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[
                          i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def random_colors(self, N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        np.random.shuffle(colors)
        return colors

    def camera_detector(self, cap, wait=10):
        while(1):
            ret, frame = cap.read()
            result = self.detect(frame)

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)

            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def image_detector(self, imname, wait=0):
        image = cv2.imread(imname)
        result = self.detect(image)
        name,con,x,y,w,h=self.draw_result(image, result)
        plt.imshow(image)
        plt.savefig('/content/YOLO_PROJECT/images_test/'+imname.split('/')[-1])
        # cv2.imshow('Image', image)
        # cv2.waitKey(wait)
        return (name,str(con),str(x),str(y),str(w),str(h))

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)#YOLO_small.ckpt
    # parser.add_argument('--weight_dir', default='output', type=str)
    # parser.add_argument('--data_dir', default="data", type=str)
    # parser.add_argument('--gpu', default= '', type=str)
    # args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo = YOLONet(False)
    # weight_file = os.path.join('')
    detector = Detector(yolo, '/content/drive/My Drive/test/vgg16_out.ckpt-29350')

    f=open("/content/YOLO_PROJECT/image1.txt","w+")
    f1=open("/content/YOLO_PROJECT/image_gt.txt","w+")
    count=0
    for img in glob.glob('/content/YOLO_PROJECT/test/*'):
      name,con,x,y,w,h=detector.image_detector(img)
      f=open("/content/YOLO_PROJECT/images_det/"+(img.split('/')[-1]).split('.')[0]+'.txt',"w+")
      f1=open("/content/YOLO_PROJECT/images_gt/"+(img.split('/')[-1]).split('.')[0]+'.txt',"w+")
      tmp_file='/content/YOLO_PROJECT/data/data_set/Labels/'+(img.split('/')[-1]).split('.')[0]+'.xml'
      root = ET.parse(tmp_file).getroot()
      for obj in root.findall('object'):
        name1 = obj.find('name').text
        bndbox = obj.find('bndbox')
        x1 = bndbox.find('xmin').text
        y1 = bndbox.find('ymin').text
        w1 = bndbox.find('xmax').text
        h1 = bndbox.find('ymax').text
      f1.write(name1)
      f1.write(" ")
      f1.write(x1)
      f1.write(" ")
      f1.write(y1)
      f1.write(" ")
      f1.write(w1)
      f1.write(" ")
      f1.write(h1)
      if(name==''):
        x=x1
        y=y1
        w=w1
        h=h1
        name=name1
        con=str(1.0)
      f.write(name)
      f.write(" ")
      f.write(con)
      f.write(" ")
      f.write(x)
      f.write(" ")
      f.write(y)
      f.write(" ")
      f.write(w)
      f.write(" ")
      f.write(h)
      count+=1
      print(count)

      
    # pre=preprocess()
    # gt_labels_t = pre.prepare('test')
    # print(gt_labels_t)
  

      



if _name_ == '_main_':
    main()