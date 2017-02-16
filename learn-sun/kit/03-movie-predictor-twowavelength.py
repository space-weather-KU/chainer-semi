#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import json, urllib, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick
import sunpy.map
from astropy.io import fits
from sunpy.cm import color_tables as ct
import sunpy.wcs as wcs
import datetime
import matplotlib.dates as mdates
import matplotlib.colors as mcol
import matplotlib.patches as ptc
from matplotlib.dates import *
import math
import scipy.ndimage.interpolation as interpolation
import subprocess
import random

from observational_data import *

import chainer
from chainer import datasets
from chainer import serializers
from chainer import links as L
from chainer import functions as F
from chainer import Variable, optimizers

image_size = 1023
image_wavelengths = [211,193]
image_wavelength = "no default"
optimizer = chainer.optimizers.SMORMS3()

dt_hours = 4

gpuid=0

def get_normalized_image_variable(time, wavelength):
    img = get_sun_image(time, wavelength)
    if img is None:
        return None
    img = img[np.newaxis, np.newaxis, :, :]
    img = img.astype(np.float32)
    x = Variable(img)
    if wavelength == 211:
        ret = F.sigmoid(x / 100)
    elif wavelength == 193:
        ret = F.sigmoid(x / 300)
    else:
        ret = F.log(F.max(1,x))
    if gpuid >= 0:
        ret.to_gpu()
    return ret



def plot_sun_image(img, filename, wavelength, title = '', vmin=0.5, vmax = 1.0):
    cmap = plt.get_cmap('sdoaia{}'.format(wavelength))
    plt.title(title)
    plt.imshow(img,cmap=cmap,origin='lower',vmin=vmin, vmax=vmax)
    plt.savefig(filename)
    plt.close("all")


class SunPredictor(chainer.Chain):
    def __init__(self):
        super(SunPredictor, self).__init__(
            # the size of the inputs to each layer will be inferred
            c1=L.Convolution2D(None,    4, 3,stride=2),
            c2=L.Convolution2D(None,    8, 3,stride=2),
            c3=L.Convolution2D(None,   16, 3,stride=2),
            c4=L.Convolution2D(None,   32, 3,stride=2),
            c5=L.Convolution2D(None,   64, 3,stride=2),
            c6=L.Convolution2D(None,  128, 3,stride=2),
            d6=L.Deconvolution2D(None, 64, 3,stride=2),
            d5=L.Deconvolution2D(None, 32, 3,stride=2),
            d4=L.Deconvolution2D(None, 16, 3,stride=2),
            d3=L.Deconvolution2D(None,  8, 3,stride=2),
            d2=L.Deconvolution2D(None,  4, 3,stride=2),
            d1=L.Deconvolution2D(None,  len(image_wavelengths), 3,stride=2)
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
        h = f(self.d6(h))
        h = f(self.d5(h))
        h = f(self.d4(h))
        h = f(self.d3(h))
        h = f(self.d2(h))
        h = F.sigmoid(self.d1(h))
        return h


model = SunPredictor()
optimizer.use_cleargrads()
optimizer.setup(model)

if gpuid >= 0:
    chainer.cuda.get_device(gpuid).use()  # Make a specified GPU current
    model.to_gpu()

epoch = 0
while True:
    epoch+=1

    vizualization_mode = (epoch%10 == 0)
    dt = datetime.timedelta(hours = dt_hours)

    t = datetime.datetime(2011,1,1,0,00,00) + datetime.timedelta(hours = random.randrange(24*365*5))
    print(epoch, t)

    img_inputs = []
    img_observeds = []
    no_image = False
    for w in image_wavelengths:
        img_input = get_normalized_image_variable(t,w)
        if img_input is None:
            no_image = True
            continue
        img_inputs.append(img_input)

        img_observed = get_normalized_image_variable(t+dt,w)
        if img_observed is None:
            no_image = True
            continue
        img_observeds.append(img_observed)

    if no_image:
        continue

    img_input = F.concat(img_inputs)
    img_observed = F.concat(img_observeds)

    img_predicted = model(img_input)

    loss = F.sqrt(F.sum((img_predicted - img_observed)**2))
    model.cleargrads()
    loss.backward()
    optimizer.update()


    if vizualization_mode:
        for c in range(len(image_wavelengths)):
            wavelength = image_wavelengths[c]

            plot_sun_image(img_input.data[0,c],
                           "aia{}-image-input.png".format(wavelength),
                           wavelength,
                           title = 'input at {}'.format(t))
            plot_sun_image(img_observed.data[0,c],
                           "aia{}-image-observed.png".format(wavelength),
                           wavelength,
                           title = 'observed at {}'.format(t+dt))

        t2 = t
        for i in range(6):
            t2 += dt

            img_input = model(img_input)

            for c in range(len(image_wavelengths)):
                wavelength = image_wavelengths[c]
                plot_sun_image(img_input.data[0,c],
                               "aia{}-image-predict-{}.png".format(wavelength, i),
                               wavelength,
                               title = 'epoch {} frame {} {}'.format(epoch, i+1, t2))

        for c in range(len(image_wavelengths)):
            wavelength = image_wavelengths[c]
            subprocess.call("convert -delay 50 aia{w}-image-input.png aia{w}-image-predict*.png aia{w}-movie.gif".format(w=wavelength), shell=True)

        serializers.save_npz('sun-predictor-{}-{}hr.model'.format(image_wavelengths, dt_hours), model)
