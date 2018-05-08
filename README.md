# 破解武汉大学图书馆系统验证码
自动识别验证码登录武汉大学图书馆系统

只使用cnn准确率达到了84%


测试图片

![avatar](test/1.png)
![avatar](test/2.png)
![avatar](test/3.png)
![avatar](test/4.png)
![avatar](test/5.png)
![avatar](test/6.png)
![avatar](test/7.png)
![avatar](test/8.png)
![avatar](test/9.png)




python3 library.py即可实现自动登录并预约

需要在config.ini文件中配置好个人信息

可以使用一个脚本每天自动预约抢座




每一个座位的ID号、开始时间、结束时间以及当前日期其实都可以爬下来

不过我懒得弄了，毕竟都大三了，也用不着图书馆了，有兴趣的同学可以继续(つД｀)･ﾟ･

关于准确率84%，我猜测是数据集大小的问题，有时间弄多点数据再训练一次(ಥ_ಥ)

嗯，就酱！
