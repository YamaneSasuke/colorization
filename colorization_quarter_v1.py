# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 13:01:00 2016

@author: yamane
"""

import os
import numpy as np
from chainer import cuda, optimizers, Chain
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
from skimage import color, io
import time


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
        h = F.relu(self.lnorm1(self.lconv1(X), test=test))
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
        y_class = self.classl2(h_class)
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
        h = F.tanh(self.onorm(self.output(h), test=test))
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
        loss_color = F.mean_squared_error(y_color, T_color)
        loss_class = F.sigmoid_cross_entropy(y_class, T_class)
        loss = loss_color + (a * loss_class)
        return loss, loss_color, loss_class

    def loss_ave(self, X, T_color, T_class, a, test):
        losses = []
        loss_colors = []
        loss_classes = []
        total_data = np.arange(len(X))
        for indexes in np.array_split(total_data, num_batches):
            X_batch = cuda.to_gpu(X[indexes])
            T_color_batch = cuda.to_gpu(T_color[indexes])
            T_class_batch = cuda.to_gpu(T_class[indexes])
            loss, loss_color, loss_class = self.lossfun(X_batch,
                                                        T_color_batch,
                                                        T_class_batch,
                                                        a, test)
            loss_cpu = cuda.to_cpu(loss.data)
            loss_color_cpu = cuda.to_cpu(loss_color.data)
            loss_class_cpu = cuda.to_cpu(loss_class.data)
            losses.append(loss_cpu)
            loss_colors.append(loss_color_cpu)
            loss_classes.append(loss_class_cpu)
        return np.mean(losses), np.mean(loss_colors), np.mean(loss_classes)

    def y_color(self, X, test):
        y_color, y_class = self.forward(X, test)
        return y_color

    def l2rgb(self, X, test):
        y_color = self.y_color(cuda.to_gpu(X), test)
        X = (X + 0.5) * 100
        X_clip = np.clip(X, 0, 100)
        y_color_cpu = cuda.to_cpu(y_color.data) * 100
        y_color_clip = np.clip(y_color_cpu, -110, 110)
        lab_bchw = np.concatenate((X_clip, y_color_clip), axis=1)
        lab_bhwc = np.transpose(lab_bchw, (0, 2, 3, 1))
        lab_bhwc_float64 = lab_bhwc.astype(np.float64)
        rgb_images = []
        for lab in lab_bhwc_float64:
            rgb_image = color.lab2rgb(lab)
            rgb_images.append(rgb_image)
        return rgb_images

if __name__ == '__main__':
    # 超パラメータ
    max_iteration = 1000  # 繰り返し回数
    learning_rate = 0.1  # 学習率
    a = 2.0
    batch_size = 100  # ミニバッチサイズ

    data_location = r'C:\Users\yamane\Dropbox\colorization\dataset'
    images = []
    losses = []
    loss_colors = []
    loss_classes = []
    class_list = []

    model = Colorizationnet().to_gpu()
    # Optimizerの設定
    optimizer = optimizers.Adam(learning_rate)
    optimizer.setup(model)

    for root, dirs, files in os.walk(data_location):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            image = io.imread(file_path)
            images.append(image)
            dirs = root.split('\\')
            dataset_index = dirs.index('resized_dataset_56')
            class_list.append(dirs[dataset_index+1])
    X = np.stack(images, axis=0)
    class_uniq = list(set(class_list))

    X_lab_bhwc = color.rgb2lab(X)
    X_lab_bchw = np.transpose(X_lab_bhwc, (0, 3, 1, 2))
    X_l_normalized = (X_lab_bchw[:, 0:1, :, :] / 100) - 0.5
    X_l_float32 = X_l_normalized.astype(np.float32)

#    low = model.low_level_features_network(X_l_gpu_float32, False)
#    gl, cl = model.global_features_network(low, False)
#    mid = model.mid_level_features_network(low, False)
#    y = model.colorization_network(mid, gl, False)
#    y_color, y_class = model.forward(X_l_gpu_float32, False)

    T_color = X_lab_bchw[:, 1:3, :, :]
    T_color = T_color.astype(np.float32)
    T_color = T_color / 50
    T_class = np.zeros((len(X), 205))
    for i in range(len(X)):
        T_class[i, class_uniq.index(class_list[i])] = 1
    T_class = T_class.astype(np.int32)

    num_batches = len(X_l_float32) / batch_size

    time_origin = time.time()
    try:

        for epoch in range(max_iteration):
            time_begin = time.time()
            permu = np.random.permutation(len(X_l_float32))
            for indexes in np.array_split(permu, num_batches):
                X_batch = cuda.to_gpu(X_l_float32[indexes])
                T_color_batch = cuda.to_gpu(T_color[indexes])
                T_class_batch = cuda.to_gpu(T_class[indexes])
                # 勾配を初期化s
                optimizer.zero_grads()
                # 順伝播を計算し、誤差と精度を取得
                loss, loss_color, loss_class = model.lossfun(X_batch,
                                                             T_color_batch,
                                                             T_class_batch,
                                                             a, False)
                # 逆伝搬を計算
                loss.backward()
                optimizer.update()

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin
            loss, loss_color, loss_class = model.loss_ave(X_l_float32,
                                                          T_color,
                                                          T_class,
                                                          a, False)
            losses.append(loss)
            loss_colors.append(loss_color)
            loss_classes.append(loss_class)
            # 訓練データでの結果を表示
            print "epoch:", epoch
            print "time", epoch_time, "(", total_time, ")"
            print "loss:", losses[epoch]
            print "loss_color:", loss_colors[epoch]
            print "loss_class:", loss_classes[epoch]
            plt.plot(losses)
            plt.plot(loss_colors)
            plt.plot(loss_classes)
            plt.title("loss")
            plt.legend(["loss", "color", "class"], loc="upper right")
            plt.grid()
            plt.show()

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

    predict_images = model.l2rgb(X_l_float32, False)
    for original_image, predict_image in zip(images[0:5],
                                             predict_images[0:5]):
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.subplot(1, 2, 2)
        plt.imshow(predict_image)
        plt.show()
