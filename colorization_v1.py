# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:19:42 2016

@author: yamane
"""

import numpy as np
from chainer import cuda, Variable, optimizers, Chain
import chainer.functions as F
import chainer.links as L
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
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
            gl1=L.Linear(25088, 1024),
            gnorm5=L.BatchNormalization(1024),
            gl2=L.Linear(1024, 512),
            gnorm6=L.BatchNormalization(512),
            gl3=L.Linear(512, 256),
            gnorm7=L.BatchNormalization(256),

            classl1=L.Linear(256, 256),
            classnorm7=L.BatchNormalization(256),
            classl2=L.Linear(256, 205),

            mconv1=L.Convolution2D(512, 512, 3, stride=1, pad=1),
            mnorm1=L.BatchNormalization(512),
            mconv2=L.Convolution2D(512, 256, 3, stride=1, pad=1),
            mnorm2=L.BatchNormalization(256),

            cconv1=L.Convolution2D(256, 128, 3, stride=1, pad=1),
            cnorm1=L.BatchNormalization(128),
            cconv2=L.Convolution2D(128, 64, 3, stride=1, pad=1),
            cnorm2=L.BatchNormalization(64),
            cconv3=L.Convolution2D(64, 64, 3, stride=1, pad=1),
            cnorm3=L.BatchNormalization(64),
            cconv4=L.Convolution2D(64, 32, 3, stride=1, pad=1),
            cnorm5=L.BatchNormalization(32),

            output=L.Convolution2D(32, 2, 3, stride=1, pad=1),
            onorm=L.BatchNormalization(2)
        )

    def low_level_features_network(self, X, test):
        x = Variable(X.reshape(-1, 1, 224, 224))
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
        h = F.relu(self.gnorm6(self.gl2(h), test=test))
        y = F.relu(self.gnorm7(self.gl3(h), test=test))
        return y

    def mid_level_features_network(self, h, test):
        h = F.relu(self.mnorm1(self.mconv1(h), test=test))
        y = F.relu(self.mnorm2(self.mconv2(h), test=test))
        return y

    def colorization_network(self, h_local, h_global, test):
        h = h_local + h_global  # 修正箇所
        h = F.relu(self.cnorm1(self.cconv1(h), test=test))
        h = F.unpooling_2d(h, 2, outsize=56)
        h = F.relu(self.cnorm2(self.cconv2(h), test=test))
        h = F.relu(self.cnorm3(self.cconv3(h), test=test))
        h = F.unpooling_2d(h, 2, outsize=112)
        h = F.relu(self.cnorm4(self.cconv4(h), test=test))
        h = F.sigmoid(self.onorm1(self.output(h), test=test))
        y = F.unpooling_2d(h, 2, outsize=224)
        return y

    def classification_network(self, X, test):
        h = self.low_level_features_network(X, test)
        h = F.relu(self.gnorm1(self.gconv1(h), test=test))
        h = F.relu(self.gnorm2(self.gconv2(h), test=test))
        h = F.relu(self.gnorm3(self.gconv3(h), test=test))
        h = F.relu(self.gnorm4(self.gconv4(h), test=test))
        h = F.relu(self.gnorm5(self.gl1(h), test=test))
        h = F.relu(self.gnorm6(self.gl2(h), test=test))
        h = F.relu(self.classnorm1(self.classl1(h), test=test))
        y = F.relu(self.classl2(h))
        return y

    def forward(self, X, test):
        h = self.low_level_features_network(X, test)
        h_local = self.mid_level_features_network(h, test)
        h_global = self.global_features_network(h, test)
        y = self.colorization_network(h_local, h_global, test)
        return y

    def lossfun(self, X, T_color, T_class, a, test):
        y_color = self.forward(X, test)
        y_class = self.classification_network(X, test)
        loss_color = F.mean_squared_error(y_color, T_color)
        loss_class = F.sigmoid_cross_entropy(y_class, T_class)
        loss = loss_color - a * loss_class
        return loss
