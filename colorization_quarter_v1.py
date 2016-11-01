# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 13:01:00 2016

@author: yamane
"""

import numpy as np
from chainer import cuda, optimizers, Chain, serializers
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
from skimage import color, io, transform
import h5py
import time
import copy
import tqdm


# ネットワークの定義
class Colorizationnet(Chain):
    def __init__(self):
        super(Colorizationnet, self).__init__(
            lconv1=L.Convolution2D(1, 32, 3, stride=2, pad=1),
            lnorm1=L.BatchNormalization(32),
            lconv2=L.Convolution2D(32, 64, 3, stride=1, pad=1),
            lnorm2=L.BatchNormalization(64),
            lconv3=L.Convolution2D(64, 64, 3, stride=2, pad=1),
            lnorm3=L.BatchNormalization(64),
            lconv4=L.Convolution2D(64, 128, 3, stride=1, pad=1),
            lnorm4=L.BatchNormalization(128),
            lconv5=L.Convolution2D(128, 128, 3, stride=2, pad=1),
            lnorm5=L.BatchNormalization(128),
            lconv6=L.Convolution2D(128, 256, 3, stride=1, pad=1),
            lnorm6=L.BatchNormalization(256),

            gconv1=L.Convolution2D(256, 256, 3, stride=2, pad=1),
            gnorm1=L.BatchNormalization(256),
            gconv2=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            gnorm2=L.BatchNormalization(256),
            gconv3=L.Convolution2D(256, 256, 3, stride=2, pad=1),
            gnorm3=L.BatchNormalization(256),
            gconv4=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            gnorm4=L.BatchNormalization(256),
            gl1=L.Linear(1024, 512),
            gnorm5=L.BatchNormalization(512),
            gl2=L.Linear(512, 256),
            gnorm6=L.BatchNormalization(256),
            gl3=L.Linear(256, 128),
            gnorm7=L.BatchNormalization(128),

            classl1=L.Linear(256, 256),
            classnorm1=L.BatchNormalization(256),
            classl2=L.Linear(256, 205),

            mconv1=L.Convolution2D(256, 256, 3, stride=1, pad=1),
            mnorm1=L.BatchNormalization(256),
            mconv2=L.Convolution2D(256, 128, 3, stride=1, pad=1),
            mnorm2=L.BatchNormalization(128),

            # num_features次元の重みをnum_classesクラス分用意する
            fusion=L.Convolution2D(256, 128, 1, stride=1),
            fnorm=L.BatchNormalization(128),
            cconv1=L.Convolution2D(128, 64, 3, stride=1, pad=1),
            cnorm1=L.BatchNormalization(64),
            cconv2=L.Convolution2D(64, 32, 3, stride=1, pad=1),
            cnorm2=L.BatchNormalization(32),
            cconv3=L.Convolution2D(32, 32, 3, stride=1, pad=1),
            cnorm3=L.BatchNormalization(32),
            cconv4=L.Convolution2D(32, 16, 3, stride=1, pad=1),
            cnorm4=L.BatchNormalization(16),

            output=L.Convolution2D(16, 2, 3, stride=1, pad=1),
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
        h_global = F.reshape(h_global, (-1, 128, 1, 1))
        h_global = F.broadcast_to(h_global, h_local.data.shape)
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
        loss_class = F.softmax_cross_entropy(y_class, T_class)
        loss = loss_color + (a * loss_class)
        return loss, loss_color, loss_class

    def loss_ave(self, image_features, T_class, num_batches, a, test):
        losses = []
        loss_colors = []
        loss_classes = []
        total_data = np.arange(num_train, len(image_features))
        for indexes in np.array_split(total_data, num_batches):
            X_batch, T_color_batch = read_images_and_T_color(image_features,
                                                             indexes)
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


def random_crop_and_flip(image_hwc, crop_size):
    image = image_hwc
    h_image, w_image = image.shape[:2]
    h_crop, w_crop = crop_size

    # 0以上 h_image - h_crop以下の整数乱数
    top = np.random.randint(0, h_image - h_crop + 1)
    left = np.random.randint(0, w_image - w_crop + 1)
    bottom = top + h_crop
    right = left + w_crop

    image = image[top:bottom, left:right]

    if np.random.rand() > 0.5:  # 半々の確率で
        image = image[:, ::-1]  # 左右反転

    return image


def read_images_and_T_color(image_features, indexes):
    images = []

    for i in indexes:
        image = image_features[i]
        image = np.transpose(image, (1, 2, 0))
        image = transform.resize(image, (64, 64))
        image = random_crop_and_flip(image, (56, 56))
        images.append(image)
    X = np.stack(images, axis=0)

    X_lab_bhwc = color.rgb2lab(X)
    X_lab_bchw = np.transpose(X_lab_bhwc, (0, 3, 1, 2))
    X_l_normalized = (X_lab_bchw[:, 0:1, :, :] / 100) - 0.5
    X_l_float32 = X_l_normalized.astype(np.float32)

    T_color = X_lab_bchw[:, 1:3, :, :]
    T_color = T_color.astype(np.float32)
    T_color = T_color / 50

    return cuda.to_gpu(X_l_float32), cuda.to_gpu(T_color)

# @profile
#def main():
if __name__ == '__main__':
    # 超パラメータ
    max_iteration = 10000  # 繰り返し回数
    learning_rate = 0.001  # 学習率
    a = 1.0 / 300.0
    batch_size = 100  # ミニバッチサイズ
    valid_size = 20000

    image_list = []
    class_list = []
    epoch_loss = []
    epoch_loss_color = []
    epoch_loss_class = []
    epoch_valid_loss = []
    epoch_valid_loss_color = []
    epoch_valid_loss_class = []
    loss_valid_best = np.inf

    model = Colorizationnet().to_gpu()
    # Optimizerの設定
    optimizer = optimizers.AdaDelta(learning_rate)
    optimizer.setup(model)

    f = open(r'E:\raw_dataset\raw_dataset_56\raw_dataset_56.txt', "r")
    for path in f:
        path = path.strip()
        dirs = path.split('\\')
        images256_index = dirs.index('images256')
        image_list.append(path)
        class_list.append('_'.join(dirs[images256_index+2:-1]))
    f.close()
    class_uniq = list(set(class_list))

#    low = model.low_level_features_network(X_l_gpu_float32, False)
#    gl, cl = model.global_features_network(low, False)
#    mid = model.mid_level_features_network(low, False)
#    y = model.colorization_network(mid, gl, False)
#    y_color, y_class = model.forward(X_l_gpu_float32, False)

    T_class = np.zeros(len(image_list))
    for i in range(len(image_list)):
        T_class[i] = class_uniq.index(class_list[i])
    T_class_train = T_class[:-valid_size].astype(np.int32)
    T_class_valid = T_class[-valid_size:].astype(np.int32)

    dataset = h5py.File(r'E:\raw_dataset\raw_dataset_56\raw_dataset_56.hdf5')
    image_features = dataset['image_features']

    num_train = len(image_features) - valid_size
    num_batches = num_train / batch_size

    time_origin = time.time()
    try:

        for epoch in range(max_iteration):
            time_begin = time.time()
            permu = range(num_train)
            losses = []
            loss_colors = []
            loss_classes = []
            for indexes in tqdm.tqdm(np.array_split(permu, num_batches)):
                X_batch, T_color_batch = read_images_and_T_color(
                        image_features, indexes)
                T_class_batch = cuda.to_gpu(T_class_train[indexes])
                # 勾配を初期化s
#                optimizer.zero_grads()
#                # 順伝播を計算し、誤差と精度を取得
#                loss, loss_color, loss_class = model.lossfun(
#                        X_batch, T_color_batch, T_class_batch, a, False)
#                # 逆伝搬を計算
#                loss.backward()
#                optimizer.update()
#                loss = cuda.to_cpu(loss.data)
#                loss_color = cuda.to_cpu(loss_color.data)
#                loss_class = cuda.to_cpu(loss_class.data)
#
#                losses.append(loss)
#                loss_colors.append(loss_color)
#                loss_classes.append(loss_class * a)

            time_end = time.time()
            epoch_time = time_end - time_begin
            total_time = time_end - time_origin
#            epoch_loss.append(np.mean(losses))
#            epoch_loss_color.append(np.mean(loss_colors))
#            epoch_loss_class.append(np.mean(loss_classes))
#
#            loss_valid, loss_color_valid, loss_class_valid = model.loss_ave(
#                    image_features, num_train, T_class_valid,
#                    num_batches, a, False)
#
#            epoch_valid_loss.append(loss_valid)
#            epoch_valid_loss_color.append(loss_color_valid)
#            epoch_valid_loss_class.append(loss_class_valid * a)
#
#            if loss_valid < loss_valid_best:
#                loss_valid_best = loss_valid
#                epoch_best = epoch
#                model_best = copy.deepcopy(model)

            # 訓練データでの結果を表示
            print "epoch:", epoch
            print "time", epoch_time, "(", total_time, ")"
#            print "loss[train]:", epoch_loss[epoch]
#            print "loss_color[train]:", epoch_loss_color[epoch]
#            print "loss_class * a[train]:", epoch_loss_class[epoch]
#            print "loss[valid]:", loss_valid
#            print "loss[valid_best]:", loss_valid_best
#            print "loss_color[valid]:", loss_color_valid
#            print "loss_class * a[valid]:", loss_class_valid * a
#
#            plt.plot(epoch_loss)
#            plt.plot(epoch_loss_color)
#            plt.plot(epoch_loss_class)
#            plt.title("loss_train")
#            plt.legend(["loss", "color", "class"], loc="upper right")
#            plt.grid()
#            plt.show()
#
#            plt.plot(epoch_valid_loss)
#            plt.plot(epoch_valid_loss_color)
#            plt.plot(epoch_valid_loss_class)
#            plt.title("loss_valid")
#            plt.legend(["loss", "color", "class"], loc="upper right")
#            plt.grid()
#            plt.show()

    except KeyboardInterrupt:
        print "割り込み停止が実行されました"

#    model_filename = 'model' + str(time.time()) + '.npz'
#    serializers.save_npz(model_filename, model_best)

    print 'max_iteration:', max_iteration
    print 'learning_rate:', learning_rate
    print 'batch_size:', batch_size
    print 'num_train', num_train
    print 'a:', a
#main()
