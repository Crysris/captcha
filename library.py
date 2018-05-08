# -*- encoding:utf:8
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import re
from cnn_captcha import CaptchaBreak
from util import convert2gray
import numpy as np
import os
import configparser

# 从配置文件中加载数据
cp = configparser.SafeConfigParser()
cp.read('config.ini')
username = cp.get('user', 'username')
password = cp.get('user', 'password')
accessHeaders = {
    'Accept':
    'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language':
    'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
    'Cache-Control':
    'max-age=0',
    'Accept-Encoding':
    'gzip, deflate',
    'User-Agent':
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36',
    'Connection':
    'Keep-Alive'
}
loginHeaders = {
    'Host':
    'seat.lib.whu.edu.cn',
    'Content-Type':
    'application/x-www-form-urlencoded',
    'User-Agent':
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36',
    'Accept':
    'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Referer':
    'http://seat.lib.whu.edu.cn/login?targetUri=%2F',
    'Accept-Encoding':
    'gzip, deflate',
    'Accept-Language':
    'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
    'Connection':
    'keep-alive',
    'Set-Cookie':
    'rememberMe=deleteMe; Path=/; Max-Age=0; Expires=Wed, 18-Apr-2018 11:13:59 GMT'
}
seatHeaders = {
    'Host':
    'seat.lib.whu.edu.cn',
    'Origin':
    'http://seat.lib.whu.edu.cn',
    'Content-Type':
    'application/x-www-form-urlencoded',
    'User-Agent':
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.139 Safari/537.36',
    'Accept':
    'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    'Referer':
    'http://seat.lib.whu.edu.cn/map',
    'Accept-Encoding':
    'gzip, deflate',
    'Accept-Language':
    'zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7',
    'Connection':
    'keep-alive'
}
url = cp.get('other', 'url')
captchaUrl = cp.get('other', 'captchaUrl')
loginUrl = cp.get('other', 'loginUrl')
mapUrl = cp.get('other', 'mapUrl')
date = cp.get('seat', 'date')
seatID = cp.get('seat', 'seatID')
start = cp.get('seat', 'start')
end = cp.get('seat', 'end')


class SeatReservation(object):
    def __init__(self):
        self.sess = requests.session()
        self.data = {'username': username, 'password': password, 'captcha': ''}
        self.capthca_break = CaptchaBreak()
        self.capthca_break.init()
        res = self.login()
        #self.soup = BeautifulSoup(res.text, 'lxml')

    def getImage(self):
        count = 1
        while count < 100000:
            try:
                captcha = self.sess.get(captchaUrl, headers=accessHeaders)
                image = BytesIO(captcha.content)
                img = Image.open(image)

                imgArray = convert2gray(np.array(img)).flatten() / 255
                captcha_text = self.capthca_break.predictByArray(imgArray)
                captcha_text = ''.join(captcha_text)
                self.data['captcha'] = captcha_text
                res = self.sess.post(
                    url=loginUrl, data=self.data, headers=loginHeaders)
                msg = re.findall('showmsg."(.*)",', res.text)
                print(msg)
                if len(msg) == 0:
                    path = 'data/' + captcha_text + '.png'
                    if not os.path.exists(path):
                        img.save(path)
                    print('img', str(count) + '.png', '预测值', captcha_text)
                count += 1
            except requests.exceptions.ConnectionError as e:
                print(e)

    def getCaptcha(self):
        captcha = self.sess.get(captchaUrl, headers=accessHeaders)
        image = BytesIO(captcha.content)
        img = Image.open(image)
        imgArray = convert2gray(np.array(img)).flatten() / 255
        captcha_text = self.capthca_break.predictByArray(imgArray)
        captcha_text = ''.join(captcha_text)
        return captcha_text

    def login(self):
        while True:
            try:
                self.data['captcha'] = self.getCaptcha()
                res = self.sess.post(
                    url=loginUrl, data=self.data, headers=loginHeaders)
                msg = re.findall('showmsg."(.*)",', res.text)
                # 登陆成功
                if len(msg) == 0:
                    print('登录成功')
                    return res

            except requests.exceptions.ConnectionError as e:
                print(e)

    def checkLoginStatus(self, res=None):
        pass

    def getSeatMap(self):
        self.sess.get(mapUrl, headers=accessHeaders)
        # soup = BeautifulSoup(res.text, 'lxml')
        getFloorUrl = 'http://seat.lib.whu.edu.cn/mapBook/ajaxGetFloor'
        data = {'id': '2'}

        self.sess.post(url=getFloorUrl, data=data, headers=accessHeaders)
        getRoomUrl = 'http://seat.lib.whu.edu.cn/mapBook/ajaxGetRooms?building=' + str(
            building) + '&floor=' + str(floor)
        # 获取教室列表
        rootInfo = self.sess.get(url=getRoomUrl, headers=accessHeaders)
        print(rootInfo.text)

    def getTime(self):
        res = self.sess.get(mapUrl, headers=accessHeaders)
        soup = BeautifulSoup(res.text)

    def reserveSeat(self):
        url1 = 'http://seat.lib.whu.edu.cn/map'
        url2 = 'http://seat.lib.whu.edu.cn/selfRes'
        data = {
            'date': date,
            'seat': seatID,
            'start': start,
            'end': end,
            'captcha': self.getCaptcha()
        }
        res = self.sess.get(url=url1, headers=accessHeaders)
        res = self.sess.post(url=url2, data=data, headers=seatHeaders)
        soup = BeautifulSoup(res.text, 'lxml')
        dt = soup.find('dt')
        dds = soup.findAll('dd')
        info = []
        info.append(dt.text)
        for dd in dds:
            info.append('\n')
            info.append(dd.text)
        info = ''.join(info)
        print(info)


seat = SeatReservation()
seat.reserveSeat()
