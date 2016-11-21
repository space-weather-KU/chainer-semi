#!/usr/bin/env python
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

image_size = 1023
image_wavelength = 1600

def get_sun_image(time, wavelength = image_wavelength):
    time_str = time.strftime("%Y.%m.%d_%H:%M:%S")

    url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?ds=aia.lev1[{}_TAI/12s][?WAVELNTH={}?]&op=rs_list&key=T_REC,CROTA2,CDELT1,CDELT2,CRPIX1,CRPIX2,CRVAL1,CRVAL2&seg=image_lev1".format(time_str, wavelength)
    response = urllib.urlopen(url)
    data = json.loads(response.read())
    filename = data['segments'][0]['values'][0]
    url = "http://jsoc.stanford.edu"+filename
    chromosphere_image = fits.open(url)   # download the data


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
    original_width = chromosphere_image[1].data.shape[0]
    return interpolation.zoom(chromosphere_image[1].data, image_size / float(original_width)) / exptime

def plot_sun_image(img, filename, wavelength=image_wavelength, title = '', vmax = 1.0):
    cmap = plt.get_cmap('sdoaia{}'.format(wavelength))
    plt.title(title)
    plt.imshow(img,cmap=cmap,origin='lower',vmin=0, vmax=vmax)
    plt.savefig(filename)
    plt.close("all")


t = datetime.datetime(2012,12,29,12,00,00)
dt = datetime.timedelta(hours = 1)

img = get_sun_image(t)
plot_sun_image(img / 100, "image-before.png", title = 'before')

img = get_sun_image(t+dt)
plot_sun_image(img / 100, "image-after.png", title = 'after')
