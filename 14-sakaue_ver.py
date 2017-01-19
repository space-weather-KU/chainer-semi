#!/usr/bin/env python2
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

import chainer
from chainer import datasets
from chainer import serializers
from chainer import links as L
from chainer import functions as F
from chainer import Variable, optimizers

image_size = 1023
image_wavelength = 1600
dt_hours = 4

def get_sun_image(time, wavelength = image_wavelength):
    try:
        time_str = time.strftime("%Y.%m.%d_%H:%M:%S")

        url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?ds=aia.lev1[{}_TAI/12s][?WAVELNTH={}?]&op=rs_list&key=T_REC,CROTA2,CDELT1,CDELT2,CRPIX1,CRPIX2,CRVAL1,CRVAL2&seg=image_lev1".format(time_str, wavelength)
        response = urllib.urlopen(url)
        data = json.loads(response.read())
        filename = data['segments'][0]['values'][0]
        url = "http://jsoc.stanford.edu"+filename
        chromosphere_image = fits.open(url, cached=False)   # download the data

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
        print e.message
        return None

def get_normalized_image_variable(time, wavelength = image_wavelength):
    img = get_sun_image(time, wavelength)
    if img is None:
        return None
    img = img[np.newaxis, np.newaxis, :, :]
    img = img.astype(np.float32)
    x = Variable(img)
    return F.sigmoid(x / 100)



def plot_sun_image(img, filename, wavelength=image_wavelength, title = '', vmin=0.5, vmax = 1.0):
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
            d1=L.Deconvolution2D(None,  1, 3,stride=2)
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
opt = chainer.optimizers.Adam()
opt.use_cleargrads()
opt.setup(model)

epoch = 0
while True:
    vizualization_mode = (epoch%10 == 0)
    dt = datetime.timedelta(hours = dt_hours)

    t = datetime.datetime(2011,1,1,0,00,00) +  datetime.timedelta(minutes = random.randrange(60*24*365*5))
    print epoch, t
    img_input = get_normalized_image_variable(t)
    if img_input is None:
        continue

    img_observed = get_normalized_image_variable(t+dt)
    if img_observed is None:
        continue

    img_predicted = model(img_input)

    loss = F.sqrt(F.sum((img_predicted - img_observed)**2))
    model.cleargrads()
    loss.backward()
    opt.update()
    serializers.save_npz('sun-predictor-{}-{}hr.model'.format(image_wavelength, dt_hours), model)

    epoch+=1
    
    subprocess.call("rm /tmp/*", shell=True)

    if vizualization_mode:
        plot_sun_image(img_input.data[0,0], "image-input.png", title = 'input at {}'.format(t))
        plot_sun_image(img_observed.data[0,0], "image-observed.png", title = 'observed at {}'.format(t+dt))

        t2 = t
        for i in range(6):
            t2 += dt
            img_input = model(img_input)
            plot_sun_image(img_input.data[0,0], "image-predict-{}.png".format(i), title = 'epoch {} frame {} {}'.format(epoch, i+1, t2))
        subprocess.call("zip images.zip -r *.png", shell=True)
        subprocess.call("base64 images.zip > images.zip.base64",shell=True)
