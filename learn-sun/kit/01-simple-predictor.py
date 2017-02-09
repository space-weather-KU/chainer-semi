#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
prediction_begin_time = datetime.datetime(2012,1,1)
current_time = prediction_begin_time

initial_learn_count = 1000
learn_per_predict = 3

wavelength = 211

gpuid=0

if "debug" in sys.argv:
    initial_learn_count = 20
    end_time = datetime.datetime(2012,1,31)    

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
    x.to_gpu()
    return x


def get_normalized_output_variable(time):
    ret = np.log10(max(1e-10,goes_max(time, datetime.timedelta(days=1))))
    x = Variable(np.array([[ret]]).astype(np.float32))
    x.to_gpu()
    return x


model = SunPredictor()
optimizer = chainer.optimizers.Adam()
optimizer.use_cleargrads()
optimizer.setup(model)

if gpuid >= 0:
    chainer.cuda.get_device(gpuid).use()  # Make a specified GPU current
    model.to_gpu()  

def save():
    try:
        serializers.save_npz("model.save", model)
        serializers.save_npz("optimizer.save", optimizer)
    except:
        pass

class PredictionItem:
    def __init__(self, t, p, o):
        self.time = t
        self.prediction = p
        self.observation = o

prediction_log = []

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

    prediction_data = float(str(prediction.data[0,0]))
    observation_data = float(str(observation.data[0,0]))

    if not training_mode:
        prediction_log.append(PredictionItem(t, prediction_data, observation_data))

    with open("log.txt", "a") as fp:
        msg = "{} {} {} {}".format(t, training_mode, prediction_data, observation_data)
        fp.write(msg + "\n")



"""
Learning logic / many thanks to Hishinuma-san
"""

wct_begin = datetime.datetime.now()

# まず、最初の1年間で練習します
for i in range(initial_learn_count):
    predict(training_mode = True)
    if i % 100 == 0:
        print("learning: ", i, "/", initial_learn_count, file=sys.stderr )
        save()

#時間をstep_timeづつ進めながら、予報実験をしていきます。
while current_time < end_time:
    predict(training_mode = False)
    if "debug" in sys.argv:
        print("predicting: ", current_time, "/", end_time , file=sys.stderr )

    for i in range(learn_per_predict):
        predict(training_mode = True)
    current_time += step_time

save()

wct_end = datetime.datetime.now()


#予報実験の結果をまとめたプロットを作ります。
def plot_history():

    plt.rcParams['figure.figsize'] = (300, 6)
    if "debug" in sys.argv:    
        plt.rcParams['figure.figsize'] = (10, 6)
    data_t = []
    data_goes_max = []
    
    t = prediction_begin_time
    while True:
        if t > end_time:
            break
        data_t.append(t) 
        data_goes_max.append(np.log10(goes_max(t, datetime.timedelta(hours=24)) ))
        t += datetime.timedelta(minutes=12)
    
    plt.plot(data_t, data_goes_max, color="b")
    data_prediction_t = [i.time for i in prediction_log]
    data_prediction_y = [i.prediction for i in prediction_log]
    plt.plot(data_prediction_t, data_prediction_y, "ro")
    
    daysFmt = mdates.DateFormatter('%Y-%m-%d')
    yearLoc = mdates.YearLocator()
    monthLoc = mdates.MonthLocator()
    plt.gca().xaxis.set_major_locator(yearLoc)
    plt.gca().xaxis.set_major_formatter(daysFmt)
    plt.gca().xaxis.set_minor_locator(monthLoc)
    plt.gca().grid()
    plt.gcf().autofmt_xdate()
    plt.gca().set_title('GOES Flux')
    plt.gca().set_xlabel('International Atomic Time')
    plt.gca().set_ylabel(u'GOES Long[1-8A] Xray Flux')
            
    plt.savefig("prediction-history.png", dpi=100)
    plt.close('all')
    
def plot_scatter():
    plt.rcParams['figure.figsize'] = (10, 10)

    # generate data
    x = [i.observation for i in prediction_log]
    y = [i.prediction for i in prediction_log]
    

    plt.gca().set_title('GOES Flux')
    plt.gca().set_xlabel('observation')
    plt.gca().set_ylabel('prediction')
    plt.gca().grid()
    plt.scatter(x,y,color="r")

    plt.savefig("prediction-observation.png", dpi=100)
    plt.close('all')
plot_history()
plot_scatter()



avg_num = 0
avg_den = 0
for i in prediction_log:
    avg_num += abs(i.prediction - i.observation)
    avg_den += 1

print("Average  error:", avg_num / avg_den)
print("Wall clock time: ", wct_end -  wct_begin)
