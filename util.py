import numpy as np
import random
# 验证码中的字符, 就不用汉字了
from captcha.image import ImageCaptcha
from PIL import Image
import os

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]
ALPHABET = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

captchaTexts = number + alphabet
MAX_CAPTCHA = 6
IMAGE_HEIGHT = 70
IMAGE_WIDTH = 160
CHAR_SET_LEN = len(captchaTexts)


# 验证码6个字符
def getRandomText():
    texts = []
    for i in range(MAX_CAPTCHA):
        c = random.choice(captchaTexts)
        texts.append(c)
    return texts


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
    captcha_text = getRandomText()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_image, captcha_text


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def getIdx(pos):
    if pos < 10:
        return pos + ord('0')
    else:
        return pos - 10 + ord('a')


def text2vec(text):
    # 把6个字符的验证码用TEXTNUM*NUMBER维的向量表示
    vec = np.zeros(CHAR_SET_LEN * MAX_CAPTCHA)

    # ord('0')=48  ord('A')=65 ord('a')=97
    def charPos(c):
        if ord(c) < 97:
            return ord(c) - ord('0')
        else:
            return ord(c) - ord('a') + 10

    for i, c in enumerate(text):
        idx = charPos(c)
        vec[idx + i * CHAR_SET_LEN] = 1
    return vec


def vec2text(vec):
    # 把向量转化为字符
    texts = []

    vec = vec.nonzero()[0]
    for i, c in enumerate(vec):
        texts.append(chr(getIdx(c % CHAR_SET_LEN)))

    return ''.join(texts)


def getTrainBatch(batch_size, trainIndexs, train_x, train_y):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    idxs = random.sample(trainIndexs, batch_size)
    for i, c in enumerate(idxs):
        image, text = train_x[c], train_y[c]
        image = convert2gray(image)
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


def getTestBatch(batch_size, testIndexs, test_x, test_y):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])
    idxs = random.sample(testIndexs, batch_size)
    for i, c in enumerate(idxs):
        image, text = test_x[c], test_y[c]
        image = convert2gray(image)
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y


# load data from local path
def loadData(data_path):
    # 数据集
    data_x = []
    data_y = []

    def readCaptcha(src=data_path):
        files = os.listdir(src)
        for file in files:
            path = os.path.join(src, file)
            if not os.path.isdir(path):
                img = Image.open(path)
                data_x.append(np.array(img))
                data_y.append(file.split('.')[0])

    readCaptcha()
    data_size = len(data_x)
    split_point = (int)(data_size * 0.9)

    # 训练集
    train_x = data_x[:split_point]
    train_y = data_y[:split_point]
    train_data_size = len(train_x)
    trainIndexs = [i for i in range(train_data_size)]

    # 测试集
    test_data_size = data_size - train_data_size
    test_x = data_x[split_point:]
    test_y = data_y[split_point:]
    testIndexs = [i for i in range(test_data_size)]
    return train_x, train_y, test_x, test_y, trainIndexs, testIndexs