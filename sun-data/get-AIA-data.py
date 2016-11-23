#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import json, urllib, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick
import sunpy.map
from astropy.io import fits
from sunpy.cm import color_tables as ct
import sunpy.wcs as wcs
from datetime import datetime as dt_obj
import matplotlib.dates as mdates
import matplotlib.colors as mcol
import matplotlib.patches as ptc
from matplotlib.dates import *
import math

url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?ds=aia.lev1[2013.03.06_23:29:06_TAI/12s][?WAVELNTH=1600?]&op=rs_list&key=T_REC,CROTA2,CDELT1,CDELT2,CRPIX1,CRPIX2,CRVAL1,CRVAL2&seg=image_lev1"
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

print(type(chromosphere_image[1].data))
print(chromosphere_image[1].data.shape)
print chromosphere_image[1].header['EXPTIME']
print chromosphere_image[1].header
