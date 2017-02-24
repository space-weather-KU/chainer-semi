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
import sys
import random

from observational_data import get_sun_image

import chainer
from chainer import datasets
from chainer import serializers
from chainer import links as L
from chainer import functions as F
from chainer import Variable, optimizers
import chainer.cuda as xp

# available wavelengths are
# 94, 131, 171, 193, 211, 304, 'hmi'


image_wavelengths = [94,211,'hmi']

optimizer_p = chainer.optimizers.SMORMS3()
optimizer_d = chainer.optimizers.SMORMS3()
optimizer_g = chainer.optimizers.SMORMS3()
start_dcgan_at_epoch=0

use_textbook_dcgan = True

dt_hours = 4

Mp=1
Mg=1
Md=1


gpuid=0
if gpuid >= 0:
    chainer.cuda.get_device(gpuid).use()


"""
Returns the brightness-normalized image of the Sun
depending on the wavelength.
"""
def get_normalized_image_variable(time, wavelength):
    img = get_sun_image(time, wavelength)
    if img is None:
        return None


    img = img[np.newaxis, np.newaxis, :, :]
    img = img.astype(np.float32)
    x = Variable(img)
    if gpuid >= 0:
        x.to_gpu()

    if wavelength == 'hmi':
        ret = x / 300
    elif wavelength == 211:
        ret = F.sigmoid(x / 100)
    elif wavelength == 193:
        ret = F.sigmoid(x / 300)
    elif wavelength == 94:
        ret = F.sigmoid(x / 30)
    else:
        ret = F.log(1+F.relu(x))

    return ret

"""
Plot the image of the sun using the
SDO-AIA map.
"""
def plot_sun_image(img, filename, wavelength, title = '', vmin=None, vmax = 1.0):
    if gpuid >= 0:
        img = img.get()


    if vmin is None:
        if wavelength == 'hmi':
            vmin = -1.0
        else:
            vmin = 0.5

    if wavelength == 'hmi':
        cmap =  plt.get_cmap('hmimag')
    else:
        cmap = plt.get_cmap('sdoaia{}'.format(wavelength))
    plt.title(title)
    plt.imshow(img,cmap=cmap,origin='lower',vmin=vmin, vmax=vmax)
    plt.savefig(filename)
    plt.close("all")


class SunPredictor(chainer.Chain):
    def __init__(self):
        super(SunPredictor, self).__init__(
            # the size of the inputs to each layer will be inferred
            c1=L.Convolution2D(None,    4*Mp, 3,stride=2),
            c2=L.Convolution2D(None,    8*Mp, 3,stride=2),
            c3=L.Convolution2D(None,   16*Mp, 3,stride=2),
            c4=L.Convolution2D(None,   32*Mp, 3,stride=2),
            c5=L.Convolution2D(None,   64*Mp, 3,stride=2),
            c6=L.Convolution2D(None,  128*Mp, 3,stride=2),
            d6=L.Deconvolution2D(None, 64*Mp, 3,stride=2),
            d5=L.Deconvolution2D(None, 32*Mp, 3,stride=2),
            d4=L.Deconvolution2D(None, 16*Mp, 3,stride=2),
            d3=L.Deconvolution2D(None,  8*Mp, 3,stride=2),
            d2=L.Deconvolution2D(None,  4*Mp, 3,stride=2),
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

class SunGenerator(chainer.Chain):
    def __init__(self):
        super(SunGenerator, self).__init__(
            # the size of the inputs to each layer will be inferred
            c1=L.Convolution2D(None,    4*Mg, 3,stride=2),
            c2=L.Convolution2D(None,    8*Mg, 3,stride=2),
            c3=L.Convolution2D(None,   16*Mg, 3,stride=2),
            c4=L.Convolution2D(None,   32*Mg, 3,stride=2),
            c5=L.Convolution2D(None,   64*Mg, 3,stride=2),
            c6=L.Convolution2D(None,  128*Mg, 3,stride=2),
            d6=L.Deconvolution2D(None, 64*Mg, 3,stride=2),
            d5=L.Deconvolution2D(None, 32*Mg, 3,stride=2),
            d4=L.Deconvolution2D(None, 16*Mg, 3,stride=2),
            d3=L.Deconvolution2D(None,  8*Mg, 3,stride=2),
            d2=L.Deconvolution2D(None,  4*Mg, 3,stride=2),
            d1=L.Deconvolution2D(None,  len(image_wavelengths), 3,stride=2)
        )

    def __call__(self, x):
        def f(x) :
            return F.elu(x)
        h0 = x
        h1 = f(self.c1(h0))
        h2 = f(self.c2(h1))
        h3 = f(self.c3(h2))
        h4 = f(self.c4(h3))
        h5 = f(self.c5(h4))
        h6 = f(self.c6(h5))
        i5 = F.concat([f(self.d6(h6)),h5])
        i4 = F.concat([f(self.d5(i5)),h4])
        i3 = F.concat([f(self.d4(i4)),h3])
        i2 = F.concat([f(self.d3(i3)),h2])
        i1 = F.concat([f(self.d2(i2)),h1])
        i0 = F.sigmoid(self.d1(i1))
        return i0

class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c1=L.Convolution2D(None,    4*Md, 3,stride=2),#511
            c2=L.Convolution2D(None,    8*Md, 3,stride=2),#255
            c3=L.Convolution2D(None,   16*Md, 3,stride=2),#127
            c4=L.Convolution2D(None,   32*Md, 3,stride=2),# 63
            c5=L.Convolution2D(None,   64*Md, 3,stride=2),# 31
            c6=L.Convolution2D(None,  128*Md, 3,stride=2),# 15
            c7=L.Convolution2D(None,  256*Md, 3,stride=2),#  7
            l1=L.Convolution2D(None,  256*Md, 1,stride=1),
            l2=L.Convolution2D(None,      1, 1,stride=1)
#            c8=L.Convolution2D(None,  512*M, 3,stride=2),
#            c9=L.Convolution2D(None, 1024*M, 3,stride=2),
#            l1=L.Linear(1024*M,1024),
#            l2=L.Linear(1024,1)
        )


    def __call__(self, x):
        def f(x) :
            return F.dropout(F.elu(x))
        h = x
        h = f(self.c1(h))
        h = f(self.c2(h))
        h = f(self.c3(h))
        h = f(self.c4(h))
        h = f(self.c5(h))
        h = f(self.c6(h))
        h = f(self.c7(h))
        h = f(self.l1(h))
        h = self.l2(h)
        return F.sum(h, axis=(2,3))

#        h = f(self.c8(h))
#        h = f(self.c9(h))
#        h = f(self.l1(h))
#        return self.l2(h)


def sigmoid_cross_entropy(x,z):
    return F.sum(F.relu(x) - x * z + F.log(1 + F.exp(-abs(x))))

predictor = SunPredictor()
optimizer_p.use_cleargrads()
optimizer_p.setup(predictor)

generator = SunGenerator()
optimizer_g.use_cleargrads()
optimizer_g.setup(generator)

discriminator = Discriminator()
optimizer_d.use_cleargrads()
optimizer_d.setup(discriminator)

if gpuid >= 0:
    predictor.to_gpu()
    generator.to_gpu()
    discriminator.to_gpu()

epoch = 0
while True:
    epoch+=1

    visualization_mode = (epoch%10 == 0) or epoch < 3
    dt = datetime.timedelta(hours = dt_hours)

    t = datetime.datetime(2011,1,1,0,00,00) + datetime.timedelta(hours = random.randrange(24*365*5))
    print(epoch, t)

    channel_inputs = []
    channel_observeds = []
    no_image = False
    for w in image_wavelengths:
        channel_input = get_normalized_image_variable(t,w)
        if channel_input is None:
            no_image = True
            continue
        channel_inputs.append(channel_input)

        channel_observed = get_normalized_image_variable(t+dt,w)
        if channel_observed is None:
            no_image = True
            continue
        channel_observeds.append(channel_observed)

    if no_image:
        continue

    img_input = F.concat(channel_inputs)
    img_observed = F.concat(channel_observeds)

    img_predicted = predictor(img_input)

    loss = F.sum(abs(img_predicted - img_observed))
    predictor.cleargrads()
    loss.backward()
    optimizer_p.update()


    """
    Train the generator and discriminator
    """
    t2 = t
    no_missing_image = True
    img_forecast = img_input
    if epoch >= start_dcgan_at_epoch :
        for i in range(1,7):
            t2 = t + i*dt
            img_forecast = predictor(img_forecast)

            channel_futures = []
            for w in image_wavelengths:
                channel_future = get_normalized_image_variable(t2,w)
                if channel_future is None:
                    no_image = True
                    continue
                channel_futures.append(channel_future)

            if no_image: # some wavelength is not available for this t2
                no_missing_image = False
                continue

            img_future = F.concat(channel_futures)

            img_generated = generator(img_forecast)

            img_po = F.concat([img_forecast, img_future])
            img_pg = F.concat([img_forecast, img_generated])

            if use_textbook_dcgan:
                loss_d = sigmoid_cross_entropy(discriminator(img_po), 0.9) + \
                         sigmoid_cross_entropy(discriminator(img_pg), 0.0)
                #loss_d = -0.9 * F.log(discriminator(img_po)) \
                #         -0.1 * F.log(1 - discriminator(img_po)) \
                #         - F.log(1 - discriminator(img_pg))
            else:
                loss_d = (discriminator(img_po)-1)**2 + (discriminator(img_pg)+1)**2
            discriminator.cleargrads()
            loss_d.backward()
            optimizer_d.update()

            if use_textbook_dcgan:
                loss_g = sigmoid_cross_entropy(discriminator(img_pg), 1.0)
                #loss_g = -F.log(discriminator(img_pg))
            else:
                loss_g = (discriminator(img_pg)-1)**2
            generator.cleargrads()
            loss_g.backward()
            optimizer_g.update()

            if visualization_mode:
                with open("log.txt","a") as fp:
                    def sample(xs):
                        return(np.min(xs), np.median(xs), np.max(xs))

                    d_op = sample(discriminator(img_po).data.get())
                    d_og = sample(discriminator(img_pg).data.get())

                    print("epoch",epoch, "range",i,
                          "L(dis)",loss_d.data.get(),
                          "L(gen)",loss_g.data.get(),
                          "D(future)",d_op,
                          "D(gen)", d_og, file=fp)


    if visualization_mode:
        for c in range(len(image_wavelengths)):
            wavelength = image_wavelengths[c]

            plot_sun_image(img_input.data[0,c],
                           "sdo{}-image-input.png".format(wavelength),
                           wavelength,
                           title = 'input at {}'.format(t))
            plot_sun_image(img_observed.data[0,c],
                           "sdo{}-image-observed.png".format(wavelength),
                           wavelength,
                           title = 'observed at {}'.format(t+dt))

        img_forecast = img_input
        for i in range(1,7):
            t2 = t + i*dt
            img_forecast = predictor(img_forecast)
            img_generated = generator(img_forecast)

            for c in range(len(image_wavelengths)):
                wavelength = image_wavelengths[c]
                plot_sun_image(img_forecast.data[0,c],
                               "sdo{}-image-predict-{}.png".format(wavelength, i),
                               wavelength,
                               title = 'epoch {} frame {} {}'.format(epoch, i+1, t2))
                plot_sun_image(img_generated.data[0,c],
                               "sdo{}-image-generated-{}.png".format(wavelength, i),
                               wavelength,
                               title = 'epoch {} frame {} {}'.format(epoch, i+1, t2))

        for c in range(len(image_wavelengths)):
            wavelength = image_wavelengths[c]
            subprocess.call("convert -delay 50 sdo{w}-image-input.png sdo{w}-image-predict*.png sdo{w}-movie-predicted.gif".format(w=wavelength), shell=True)
            subprocess.call("convert -delay 50 sdo{w}-image-input.png sdo{w}-image-generated*.png sdo{w}-movie-generated.gif".format(w=wavelength), shell=True)

        serializers.save_npz('sun-predictor-{}-{}hr.save'.format(image_wavelengths, dt_hours), predictor)
