import numpy as np
import tensorflow as tf
from PIL import Image
from util import CHAR_SET_LEN
from util import MAX_CAPTCHA
from util import IMAGE_HEIGHT
from util import IMAGE_WIDTH
from util import loadData
from util import getTestBatch
from util import getTrainBatch
from util import text2vec
from util import getIdx
from util import vec2text
from util import convert2gray
import os

model_path = 'mymodel'
tensorboard_dir = 'tensorboard'
data_path = '/Users/chris/Documents/DataSets/WHU_captcha/data'


class CaptchaBreak(object):
    def weightVariable(self, shape):
        weights = tf.truncated_normal(shape=shape, stddev=0.01)
        return tf.Variable(weights, name='w')

    def biasVariable(self, shape):
        bias = tf.constant(0.01, shape=shape)
        return tf.Variable(bias, name='b')

    def conv2d(self, x, w, stride):
        return tf.nn.conv2d(
            x, w, strides=[1, stride, stride, 1], padding='SAME')

    def maxpool2x2(self, x):
        return tf.nn.max_pool(
            x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def writeHistogram(self, weights, bias, act):
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('bias', bias)
        tf.summary.histogram('activations', act)

    def iniNetwork(self):
        '''输入图像大小为[60,170,1]'''
        with tf.name_scope('conv1'):
            weights_conv1 = self.weightVariable([3, 3, 1, 32])
            bias_conv1 = self.biasVariable([32])
            h_conv1 = tf.nn.relu(
                tf.add(self.conv2d(self._X, weights_conv1, 1), bias_conv1))
            self.writeHistogram(weights_conv1, bias_conv1, h_conv1)

        with tf.name_scope('max_pool1'):
            h_pool1 = self.maxpool2x2(h_conv1)

        with tf.name_scope('dropout1'):
            h_pool1_drop = tf.nn.dropout(h_pool1, self._prob)

        with tf.name_scope('conv2'):
            weights_conv2 = self.weightVariable([3, 3, 32, 64])
            bias_conv2 = self.biasVariable([64])
            h_conv2 = tf.nn.relu(
                tf.add(
                    self.conv2d(h_pool1_drop, weights_conv2, 1), bias_conv2))
            self.writeHistogram(weights_conv2, bias_conv2, h_conv2)

        with tf.name_scope('maxpool2'):
            h_pool2 = self.maxpool2x2(h_conv2)

        with tf.name_scope('dropout2'):
            h_pool2_drop = tf.nn.dropout(h_pool2, self._prob)

        with tf.name_scope('conv3'):
            weights_conv3 = self.weightVariable([3, 3, 64, 64])
            bias_conv3 = self.biasVariable([64])
            h_conv3 = tf.nn.relu(
                tf.add(
                    self.conv2d(h_pool2_drop, weights_conv3, 1), bias_conv3))

            self.writeHistogram(weights_conv3, bias_conv3, h_conv3)

        with tf.name_scope('maxpool3'):
            h_pool3 = self.maxpool2x2(h_conv3)

        with tf.name_scope('dropout3'):
            h_pool3_drop = tf.nn.dropout(h_pool3, self._prob)

        with tf.name_scope('reshape'):
            h_pool3_drop_flat = tf.reshape(h_pool3_drop, [-1, 9 * 20 * 64])
        with tf.name_scope('fc1'):
            weights_fc1 = self.weightVariable([9 * 20 * 64, 1024])
            bias_fc1 = self.biasVariable([1024])
            h_fc1 = tf.add(tf.matmul(h_pool3_drop_flat, weights_fc1), bias_fc1)
            h_fc1 = tf.nn.relu(h_fc1)

        with tf.name_scope('fc1_drop'):
            h_fc1_drop = tf.nn.dropout(h_fc1, self._prob)

        with tf.name_scope('fc2_readout'):
            weights_fc2 = self.weightVariable(
                [1024, MAX_CAPTCHA * CHAR_SET_LEN])
            bias_fc2 = self.biasVariable([MAX_CAPTCHA * CHAR_SET_LEN])
            readout = tf.add(tf.matmul(h_fc1_drop, weights_fc2), bias_fc2)

        return readout, h_conv1, h_conv2, h_conv3

    def init(self):
        '''验证码字符数为NUMBER'''
        self.sess = tf.InteractiveSession()

        # 神经网络输入
        with tf.name_scope('x_input'):
            self._x = tf.placeholder(
                tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH], name='x')
        with tf.name_scope('y_input'):
            self._y = tf.placeholder(
                tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN], name='label')
        with tf.name_scope('x_reshape'):
            self._X = tf.reshape(
                self._x, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

        with tf.name_scope('keep_prob'):
            self._prob = tf.placeholder(tf.float32, name='keep_prob')

        # 神经网络输出
        #with tf.name_scope('readout_and_convs'):
        self.readout, self.conv1, self.conv2, self.conv3 = self.iniNetwork()

        # 计算交叉熵
        with tf.name_scope('cross_entropy'):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self._y, logits=self.readout))

        # 预测值
        #with tf.name_scope('predict'):
        self.pred = tf.argmax(
            tf.reshape(self.readout, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)

        # 准确率
        with tf.name_scope('accuracy'):
            label = tf.argmax(
                tf.reshape(self._y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
            correct_predict = tf.equal(self.pred, label)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))

        # 定义优化器
        with tf.name_scope('optimizer'):
            self.train_step = tf.train.AdamOptimizer(0.001).minimize(
                self.cross_entropy)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(model_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('model loaded successfully!',
                  checkpoint.model_checkpoint_path)
        else:
            print('model loaded failed!')

        # 存储交叉熵误差
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        # 存储准确率
        tf.summary.scalar('accuracy', self.accuracy)
        # 存储输入图像
        tf.summary.image('input', self._X, 3)
        # 收集以上所有信息
        self.merged_summary = tf.summary.merge_all()
        # tensorboard计算图保存目录
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)

        self.writer = tf.summary.FileWriter(tensorboard_dir)
        # 保存计算图
        #self.writer.add_graph(self.sess.graph)

    def train(self):
        train_x, train_y, test_x, test_y, trainIndexs, testIndexs = loadData()

        epoch = 0
        EPOCHS = 20000
        while epoch <= EPOCHS:
            batch_image, batch_text = getTrainBatch(64)
            _, loss = self.sess.run(
                [self.train_step, self.cross_entropy],
                feed_dict={
                    self._x: batch_image,
                    self._y: batch_text,
                    self._prob: 0.75
                })
            print('epoch', epoch, '/loss', loss)

            if epoch % 100 == 0:
                test_batch_image, test_batch_text = getTestBatch(100)
                accu = self.accuracy.eval(
                    feed_dict={
                        self._x: test_batch_image,
                        self._y: test_batch_text,
                        self._prob: 1
                    })
                s = self.merged_summary.eval(
                    feed_dict={
                        self._x: test_batch_image,
                        self._y: test_batch_text,
                        self._prob: 1
                    })
                self.writer.add_summary(s, epoch)
                print('epoch', epoch, '/accuracy', accu)
                if accu >= 1.0:
                    break

            epoch += 1
        self.saver.save(self.sess, model_path + '/', global_step=epoch)

    def predictaByPath(self, img):
        '''通过图片路径来预测验证码'''
        img = np.array(Image.open(img))
        img = convert2gray(img).flatten() / 255
        pred = self.pred.eval(feed_dict={self._x: [img], self._prob: 1})

        captcha_text = []
        for i, c in enumerate(pred):
            for j in range(MAX_CAPTCHA):
                captcha_text.append(chr(getIdx(c[j])))

        return captcha_text

    def predictByArray(self, img):
        pred = self.pred.eval(feed_dict={self._x: [img], self._prob: 1})
        captcha_text = []
        for i, c in enumerate(pred):
            for j in range(MAX_CAPTCHA):
                captcha_text.append(chr(getIdx(c[j])))

        return captcha_text

    def test(self, src='test/'):
        for i in range(4632, 4638):
            text = self.predictaByPath(src + str(i) + '.png')
            print('图片', src + str(i) + '.png 预测输入：', text)


'''captcha_break = CaptchaBreak()
captcha_break.init()
captcha_break.test()'''
#captcha_break.train()'
#captcha_break.evaluate()
