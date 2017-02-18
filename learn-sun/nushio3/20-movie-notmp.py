#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import json, urllib, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick, sys
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


import chainer
from chainer import datasets
from chainer import serializers
from chainer import links as L
from chainer import functions as F
from chainer import Variable, optimizers
import chainer.cuda as xp

image_wavelengths = [211]
optimizer_p = chainer.optimizers.SMORMS3()
optimizer_d = chainer.optimizers.SMORMS3()
optimizer_g = chainer.optimizers.SMORMS3()
start_dcgan_at_epoch=0

image_size = 1023

dt_hours = 4

gpuid=-1
if gpuid >= 0:
    chainer.cuda.get_device(gpuid).use()

image_size = 1023
def get_sun_image(time, wavelength):
    try:
        time_str = time.strftime("%Y.%m.%d_%H:%M:%S")

        url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?ds=aia.lev1[{}_TAI/12s][?WAVELNTH={}?]&op=rs_list&key=T_REC,CROTA2,CDELT1,CDELT2,CRPIX1,CRPIX2,CRVAL1,CRVAL2&seg=image_lev1".format(time_str, wavelength)
        response = urllib.request.urlopen(url)
        data = json.loads(response.read().decode())
        filename = data['segments'][0]['values'][0]
        url = "http://jsoc.stanford.edu"+filename
        chromosphere_image = fits.open(url, cached="debug" in sys.argv)   # download the data

        T_REC = data['keywords'][0]['values'][0]
        CROTA2_AIA = float(data['keywords'][1]['values'][0])
        CDELT1_AIA = float(data['keywords'][2]['values'][0])
        CDELT2_AIA = float(data['keywords'][3]['values'][0])
        CRPIX1_AIA = float(data['keywords'][4]['values'][0])
        CRPIX2_AIA = float(data['keywords'][5]['values'][0])
        CRVAL1_AIA = float(data['keywords'][6]['values'][0])
        CRVAL2_AIA = float(data['keywords'][7]['values'][0])


        chromosphere_image.verify("fix")
        exptime = chromosphere_image[1].header['EXPTIME']
        if exptime <= 0:
            return None

        original_width = chromosphere_image[1].data.shape[0]
        return interpolation.zoom(chromosphere_image[1].data, image_size / float(original_width)) / exptime
    except Exception as e:
        print(e)
        return None


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

    if wavelength == 211:
        ret = F.sigmoid(x / 100)
    elif wavelength == 193:
        ret = F.sigmoid(x / 300)
    elif wavelength == 94:
        ret = F.sigmoid(x / 30)
    else:
        ret = F.log(F.max(1,x))

    return ret

"""
Plot the image of the sun using the
SDO-AIA map.
"""
def plot_sun_image(img, filename, wavelength, title = '', vmin=0.5, vmax = 1.0):
    if gpuid >= 0:
        img = img.get()

    cmap = plt.get_cmap('sdoaia{}'.format(wavelength))
    plt.title(title)
    plt.imshow(img,cmap=cmap,origin='lower',vmin=vmin, vmax=vmax)
    plt.savefig(filename)
    plt.close("all")



M=2
class SunPredictor(chainer.Chain):
    def __init__(self):
        super(SunPredictor, self).__init__(
            # the size of the inputs to each layer will be inferred
            c1=L.Convolution2D(None,    4*M, 3,stride=2),
            c2=L.Convolution2D(None,    8*M, 3,stride=2),
            c3=L.Convolution2D(None,   16*M, 3,stride=2),
            c4=L.Convolution2D(None,   32*M, 3,stride=2),
            c5=L.Convolution2D(None,   64*M, 3,stride=2),
            c6=L.Convolution2D(None,  128*M, 3,stride=2),
            d6=L.Deconvolution2D(None, 64*M, 3,stride=2),
            d5=L.Deconvolution2D(None, 32*M, 3,stride=2),
            d4=L.Deconvolution2D(None, 16*M, 3,stride=2),
            d3=L.Deconvolution2D(None,  8*M, 3,stride=2),
            d2=L.Deconvolution2D(None,  4*M, 3,stride=2),
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

class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c1=L.Convolution2D(None,    4*M, 3,stride=2),#511
            c2=L.Convolution2D(None,    8*M, 3,stride=2),#255
            c3=L.Convolution2D(None,   16*M, 3,stride=2),#127
            c4=L.Convolution2D(None,   32*M, 3,stride=2),# 63
            c5=L.Convolution2D(None,   64*M, 3,stride=2),# 31
            c6=L.Convolution2D(None,  128*M, 3,stride=2),# 15
            c7=L.Convolution2D(None,  256*M, 3,stride=2),#  7
            l1=L.Convolution2D(None,  256*M, 1,stride=1),
            l2=L.Convolution2D(None,      1, 1,stride=1)
#            c8=L.Convolution2D(None,  512*M, 3,stride=2),
#            c9=L.Convolution2D(None, 1024*M, 3,stride=2),
#            l1=L.Linear(1024*M,1024),
#            l2=L.Linear(1024,1)
        )


    def __call__(self, x):
        def f(x) :
            return F.dropout(F.leaky_relu(x))
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

generator = SunPredictor()
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
    # subprocess.call("rm /tmp/*", shell=True)


    epoch+=1

    visualization_mode = (epoch%10 == 0) or epoch < 3
    dt = datetime.timedelta(hours = dt_hours)

    t = datetime.datetime(2011,1,1,0,00,00) + datetime.timedelta(minutes = random.randrange(60*24*365*5))
    if "debug" in sys.argv:
        t = datetime.datetime(2011,1,1,0,00,00)

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

            img_op = F.concat([img_forecast, img_future])
            img_og = F.concat([img_forecast, img_generated])

            loss_d = sigmoid_cross_entropy(discriminator(img_op), 0.9) + \
                     sigmoid_cross_entropy(discriminator(img_og), 0.0)
            discriminator.cleargrads()
            loss_d.backward()
            optimizer_d.update()

            loss_g = sigmoid_cross_entropy(discriminator(img_og), 1.0)

            generator.cleargrads()
            loss_g.backward()
            optimizer_g.update()

            if visualization_mode:
                with open("log.txt","a") as fp:
                    def sample(xs):
                        return(np.min(xs), np.median(xs), np.max(xs))

                    d_op = sample(discriminator(img_op).data)
                    d_og = sample(discriminator(img_og).data)

                    print("epoch",epoch, "range",i,
                          "L(dis)",loss_d.data,
                          "L(gen)",loss_g.data,
                          "D(future)",d_op,
                          "D(gen)", d_og, file=fp)


    if visualization_mode:
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

        img_forecast = img_input
        for i in range(1,7):
            t2 = t + i*dt
            img_forecast = predictor(img_forecast)
            img_generated = generator(img_forecast)

            for c in range(len(image_wavelengths)):
                wavelength = image_wavelengths[c]
                plot_sun_image(img_forecast.data[0,c],
                               "aia{}-image-predict-{}.png".format(wavelength, i),
                               wavelength,
                               title = 'epoch {} frame {} {}'.format(epoch, i+1, t2))
                plot_sun_image(img_generated.data[0,c],
                               "aia{}-image-generated-{}.png".format(wavelength, i),
                               wavelength,
                               title = 'epoch {} frame {} {}'.format(epoch, i+1, t2))

        for c in range(len(image_wavelengths)):
            wavelength = image_wavelengths[c]
            subprocess.call("convert -delay 50 aia{w}-image-input.png aia{w}-image-predict*.png aia{w}-movie-predicted.gif".format(w=wavelength), shell=True)
            subprocess.call("convert -delay 50 aia{w}-image-input.png aia{w}-image-generated*.png aia{w}-movie-generated.gif".format(w=wavelength), shell=True)

        serializers.save_npz('sun-predictor-{}-{}hr.save'.format(image_wavelengths, dt_hours), predictor)
