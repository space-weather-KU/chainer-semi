import observational_data

import datetime, math,os, random,scipy.ndimage, sys, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import pylab
from astropy.io import fits
from observational_data import *

import chainer
from chainer import Variable
from chainer import computational_graph
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers


"""
Learning parameters
"""

begin_time = datetime.datetime(2011,1,1)
end_time = datetime.datetime(2016,1,1)
step_time = datetime.timedelta(days=1)
current_time = datetime.datetime(2012,1,1)

initial_learn_count = 1000
learn_per_predict = 3

wavelength = 211


"""
The model
"""

class SunPredictor(chainer.Chain):
    def __init__(self):
        super(SunPredictor, self).__init__(
            # the size of the inputs to each layer will be inferred,
            c1=L.Convolution2D(None,    4, 3,stride=2), # the size of image :511
            c2=L.Convolution2D(None,    8, 3,stride=2), # 255
            c3=L.Convolution2D(None,   16, 3,stride=2), # 127
            c4=L.Convolution2D(None,   32, 3,stride=2), #  63
            c5=L.Convolution2D(None,   64, 3,stride=2), #  31
            c6=L.Convolution2D(None,  128, 3,stride=2), #  15
            c7=L.Convolution2D(None,  256, 3,stride=2), #   7
            c8=L.Convolution2D(None,  512, 3,stride=2), #   3
            c9=L.Convolution2D(None, 1024, 3,stride=2), #   1
            l1=L.Linear(1024,1024),
            l2=L.Linear(1024,1)
        )


    def __call__(self, x):
        def f(x) :
            return F.elu(x)
        h = x
        h = f(self.c1(h))
        h = f(self.c2(h))
        h = f(self.c3(h))
        h = f(self.c4(h))
        h = f(self.c5(h))
        h = f(self.c6(h))
        h = f(self.c7(h))
        h = f(self.c8(h))
        h = f(self.c9(h))
        h = f(self.l1(h))
        return self.l2(h)

def get_normalized_image_variable(time, wavelength):
    img = get_sun_image(time, wavelength)
    if img is None:
        return None
    img = img[np.newaxis, np.newaxis, :, :]
    img = img.astype(np.float32)
    x = Variable(np.log(np.maximum(1,img)))
    return x

def get_normalized_output_variable(time):
    ret = np.log10(max(1e-10,goes_max(time, datetime.timedelta(days=1))))
    return Variable(np.array([[ret]]).astype(np.float32))


model = SunPredictor()
optimizer = chainer.optimizers.Adam()
optimizer.use_cleargrads()
optimizer.setup(model)

def save():
    try:
        serializers.save_npz("model.save", model)
        serializers.save_npz("optimizer.save", optimizer)
    except:
        pass


def predict(training_mode):
    if not training_mode:
        t = current_time
    else:
        learning_span = current_time - begin_time - datetime.timedelta(hours=24)
        while True:
            t = begin_time + random.random() * learning_span
            t2 = datetime.datetime(t.year, t.month,t.day, t.hour)
            if t2 < begin_time + learning_span:
                t = t2
                break

    img = get_normalized_image_variable(t, wavelength)
    if img is None:
        return
    observation = get_normalized_output_variable(t)

    prediction = model(img)
    model.cleargrads()

    loss = F.sqrt(F.sum(prediction-observation) ** 2)

    if training_mode:
        loss.backward()
        optimizer.update()

    with open("log.txt", "a") as fp:
        msg = "{} {} {} {}".format(t, training_mode, prediction.data[0,0], observation.data[0,0])
        fp.write(msg + "\n")



"""
Learning logic / many thanks to Hishinuma-san
"""



# まず、最初の1年間で練習します
for i in range(initial_learn_count):
    predict(training_mode = True)
    if i % 100 == 0:
        print("learning: ", i, "/", initial_learn_count )
        save()

#時間をpredit_step_hour時間づつ進めながら、予報実験をしていきます。
while current_time < end_time:
    predict(training_mode = False)
    print("predicting: ", t, "/", predict_count)

    for i in range(learn_per_predict):
        predict(training_mode = True)
    current_time += step_time


