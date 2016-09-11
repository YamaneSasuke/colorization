# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 13:01:00 2016

@author: yamane
"""

import numpy as np
from chainer import cuda, Variable, optimizers, Chain
import chainer.functions as F
import chainer.links as L
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from skimage import color, data, io
import glob
import os


# ネットワークの定義
class Colorizationnet(Chain):
    def __init__(self):
        super(Colorizationnet, self).__init__(
            lconv1=L.Convolution2D(1, 64, 3, stride=2, pad=1),
            lnorm1=L.BatchNormalization(64),
            lconv2=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            lnorm2=L.BatchNormalization(128),
            lconv3=L.Convolution2D(128, 128, 3, stride=2, pad=1),
            lnorm3=L.BatchNormalization(128),
            lconv4=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            lnorm4=L.BatchNormalization(256),
            lconv5=L.Convolution2D(256, 256, 3, stride=2, pad=1),
            lnorm5=L.BatchNormalization(256),
            lconv6=L.Convolution2D(256, 512, 3, stride=1, pad=1),
            lnorm6=L.BatchNormalization(512),

            gconv1=L.Convolution2D(512, 512, 3, stride=2, pad=1),
            gnorm1=L.BatchNormalization(512),
            gconv2=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            gnorm2=L.BatchNormalization(512),
            gconv3=L.Convolution2D(512, 512, 3, stride=2, pad=1),
            gnorm3=L.BatchNormalization(512),
            gconv4=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            gnorm4=L.BatchNormalization(512),
            gl1=L.Linear(2048, 1024),
            gnorm5=L.BatchNormalization(1024),
            gl2=L.Linear(1024, 512),
            gnorm6=L.BatchNormalization(512),
            gl3=L.Linear(512, 256),
            gnorm7=L.BatchNormalization(256),

            classl1=L.Linear(512, 256),
            classnorm1=L.BatchNormalization(256),
            classl2=L.Linear(256, 205),

            mconv1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            mnorm1=L.BatchNormalization(512),
            mconv2=L.Convolution2D(512, 256, 3, stride=1, pad=1),
            mnorm2=L.BatchNormalization(256),

            # num_features次元の重みをnum_classesクラス分用意する
            fusion=L.Convolution2D(512, 256, 1, stride=1),
            fnorm=L.BatchNormalization(256),
            cconv1=L.Convolution2D(256, 128, 3, stride=1, pad=1),
            cnorm1=L.BatchNormalization(128),
            cconv2=L.Convolution2D(128, 64, 3, stride=1, pad=1),
            cnorm2=L.BatchNormalization(64),
            cconv3=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            cnorm3=L.BatchNormalization(64),
            cconv4=L.Convolution2D(64, 32, 3, stride=1, pad=1),
            cnorm4=L.BatchNormalization(32),

            output=L.Convolution2D(32, 2, 3, stride=1, pad=1),
            onorm=L.BatchNormalization(2)
        )

    def low_level_features_network(self, X, test):
        x = Variable(X.reshape(-1, 1, 56, 56))
        h = F.relu(self.lnorm1(self.lconv1(x), test=test))
        h = F.relu(self.lnorm2(self.lconv2(h), test=test))
        h = F.relu(self.lnorm3(self.lconv3(h), test=test))
        h = F.relu(self.lnorm4(self.lconv4(h), test=test))
        h = F.relu(self.lnorm5(self.lconv5(h), test=test))
        y = F.relu(self.lnorm6(self.lconv6(h), test=test))
        return y

    def global_features_network(self, h, test):
        h = F.relu(self.gnorm1(self.gconv1(h), test=test))
        h = F.relu(self.gnorm2(self.gconv2(h), test=test))
        h = F.relu(self.gnorm3(self.gconv3(h), test=test))
        h = F.relu(self.gnorm4(self.gconv4(h), test=test))
        h = F.relu(self.gnorm5(self.gl1(h), test=test))
        h_global = F.relu(self.gnorm6(self.gl2(h), test=test))
        h_class = F.relu(self.classnorm1(self.classl1(h_global), test=test))
        y_class = F.relu(self.classl2(h_class))
        y_global = F.relu(self.gnorm7(self.gl3(h_global), test=test))
        return y_global, y_class

    def mid_level_features_network(self, h, test):
        h = F.relu(self.mnorm1(self.mconv1(h), test=test))
        y = F.relu(self.mnorm2(self.mconv2(h), test=test))
        return y

    def colorization_network(self, h_local, h_global, test):
        h_global.data = h_global.data.reshape(-1, 256, 1, 1)
        h_global = cuda.to_cpu(h_global.data)
#        h_local = cuda.to_cpu(h_local)
        h_global = np.broadcast_to(h_global, h_local.data.shape)
        h_global = cuda.to_gpu(h_global)
        h = F.concat((h_local.data, h_global), axis=1)
        h = F.relu(self.fnorm(self.fusion(h), test=test))
        h = F.relu(self.cnorm1(self.cconv1(h), test=test))
        h = F.unpooling_2d(h, 2, outsize=[14, 14])
        h = F.relu(self.cnorm2(self.cconv2(h), test=test))
        h = F.relu(self.cnorm3(self.cconv3(h), test=test))
        h = F.unpooling_2d(h, 2, outsize=[28, 28])
        h = F.relu(self.cnorm4(self.cconv4(h), test=test))
        h = F.sigmoid(self.onorm(self.output(h), test=test))
        y = F.unpooling_2d(h, 2, outsize=[56, 56])
        return y

    def forward(self, X, test):
        h = self.low_level_features_network(X, test)
        h_local = self.mid_level_features_network(h, test)
        h_global, y_class = self.global_features_network(h, test)
        y = self.colorization_network(h_local, h_global, test)
        return y, y_class

    def lossfun(self, X, T_color, T_class, a, test):
        y_color, y_class = self.forward(X, test)
        loss_color = F.mean_squared_error(y_color[0], T_color)
        loss_class = F.sigmoid_cross_entropy(y_class[0], T_class)
        loss = loss_color + a * loss_class
        return loss

    def y_color(self, X, test):
        h = self.low_level_features_network(X, test)
        h_local = self.mid_level_features_network(h, test)
        h_global, y_class = self.global_features_network(h, test)
        y = self.colorization_network(h_local, h_global, test)
        return y

    def predict(self, X, test):
        y_color = self.y_color(X, test)
        y_color = y_color.data[0]
        gray_rgb = color.gray2rgb(X)
        gray_lab = color.rgb2lab(gray_rgb)

        for h in range(56):
            for w in range(56):
                for c in range(1, 3):
                    gray_lab[h][w][c] = y_color[c-1][h][w]

        predict_image = color.lab2rgb(gray_lab)
        io.imshow(predict_image)

if __name__ == '__main__':
    # 超パラメータ
    max_iteration = 1600  # 繰り返し回数
    batch_size = 100  # ミニバッチサイズ
    learning_rate = 0.001  # 学習率
    a = 0

    model = Colorizationnet().to_gpu()
    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    images = []
    path = r"C:\Users\yamane\Dropbox\colorization\dataset"
    file_list = glob.glob(os.path.join(path, '*.jpg'))
    for image_path in file_list:
        image = io.imread(image_path)
        images.append(image)
    X = np.stack(images, axis=0)

#    image = io.imread("sample56.jpg")
    X_gray = color.rgb2gray(X)
    X_lab = color.rgb2lab(X)
    X_lab = np.transpose(X_lab, (0, 3, 1, 2))
    X_gray_gpu = cuda.to_gpu(X_gray)
    X_gray_gpu = X_gray_gpu.astype(np.float32)

    low = model.low_level_features_network(X_gray_gpu, False)
    gl, cl = model.global_features_network(low, False)
    mid = model.mid_level_features_network(low, False)
    y = model.colorization_network(mid, gl, False)
    y_color, y_class = model.forward(X_gray_gpu, False)

    t_color = cuda.to_gpu(X_lab[:, 1:, :, :])
    t_color = t_color.astype(np.float32)
    t_class = np.zeros(205)
    t_class = cuda.to_gpu(t_class.astype(np.int32))

    losses = []
    try:

        for epoch in range(max_iteration):
            # 勾配を初期化
            optimizer.zero_grads()
            # 順伝播を計算し、誤差と精度を取得
            loss = model.lossfun(X_gray_gpu, t_color, t_class, a, False)
            # 逆伝搬を計算
            loss.backward()
            optimizer.update()
            loss.data = cuda.to_cpu(loss.data)
            losses.append(loss.data)
            # 訓練データでの結果を表示
            print "epoch:", epoch
            print "loss:", losses[epoch]
            plt.plot(losses)
            plt.title("loss")
            plt.grid()
            plt.show()

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

    model.predict(X_gray_gpu, False)
