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
start_dcgan_at_epoch=1000

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
        aia_image = fits.open(url, cached="debug" in sys.argv)   # download the data

        aia_image.verify("fix")
        exptime = aia_image[1].header['EXPTIME']
        if exptime <= 0:
            return None
        quality = aia_image[1].header['QUALITY']
        if quality !=0:
            print(time, "bad quality",file=sys.stderr)
            return None

        original_width = aia_image[1].data.shape[0]
        return interpolation.zoom(aia_image[1].data, image_size / float(original_width)) / exptime
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


t = datetime.datetime(2015,1,1)


for ctr  in range(6):
    t1 = t + ctr * datetime.timedelta(seconds = 12)
    img = get_normalized_image_variable(t1, 211)
    if img is None:
        print("image missing: ", t1)
    plot_sun_image(img.data[0,0], "slowmotion-{}.png".format(ctr), 211, str(t1))
