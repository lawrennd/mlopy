#!/usr/bin/env python
from netlab import *
import numpy as np
import matplotlib.pyplot as pp

#"""DEMGAUSS Demonstrate sampling from Gaussian distributions.
#
#	Description
#
#	DEMGAUSS provides a simple illustration of the generation of data
#	from Gaussian distributions. It first samples from a one-dimensional
#	distribution using RANDN, and then plots a normalized histogram
#	estimate of the distribution using HISTP together with the true
#	density calculated using GAUSS.
#
#	DEMGAUSS then demonstrates sampling from a Gaussian distribution in
#	two dimensions. It creates a mean vector and a covariance matrix, and
#	then plots contours of constant density using the function GAUSS. A
#	sample of points drawn from this distribution, obtained using the
#	function GSAMP, is then superimposed on the contours.
#
#	See also
#	GAUSS, GSAMP, HISTP
#

#	Copyright (c) Ian T Nabney (1996-2001)
#        and Neil D. Lawrence (2009) (translation to python)"""

clc()
mean = np.array([2]); var = np.array([[5]]); nsamp = 3000
xmin = -10; xmax = 10; nbins = 30
print "Demonstration of sampling from a uni-variate Gaussian with mean"
dstring = str(mean) + ' and variance ' + str(var) + '.  ' + str(nsamp) + ' samples are taken.'
print dstring
x = mean + np.sqrt(var)*np.random.randn(nsamp, 1)
fh1 = pp.figure()
histp(x, xmin, xmax, nbins)
pp.hold(True)
pp.axis([xmin, xmax, 0, 0.2])
plotvals = np.linspace(xmin, xmax, 200).reshape((-1, 1))
probs = gauss(mean, var, plotvals)
pp.plot(plotvals, probs, '-r')
pp.xlabel('X')
pp.ylabel('Density')

print " "
raw_input('Press any key to continue')

mu = np.array([3, 2])
lam1 = 0.5
lam2 = 5.0
Sigma = lam1*np.ones((2, 2)) + lam2*np.array([[1, -1], [-1, 1]])
print " "
print "Demonstration of sampling from a bi-variate Gaussian.  The mean is"
dstring = '[' + str(mu[0]) + ', ' + str(mu[1]) + '] and the covariance matrix is'
print dstring
print Sigma
ngrid = 40
cmin = -5.0; cmax = 10.0
step = (cmax-cmin)/float(ngrid)
cvals = np.r_[cmin:cmax:step]
X1, X2 = np.mgrid[cmin:cmax:step, cmin:cmax:step]
XX = np.c_[X1.flatten(), X2.flatten()]
probs = gauss(mu, Sigma, XX)
probs = np.reshape(probs, (ngrid, ngrid))

fh2 = pp.figure()
pp.contour(X1, X2, probs)
pp.hold(True)

nsamp = 300
dstring = str(nsamp) + ' samples are generated.'
print "The plot shows the sampled data points with a contour plot of their density."
samples = gsamp(mu, Sigma, nsamp)
pp.plot(samples[:,0], samples[:,1], 'or')
pp.xlabel('X1')
pp.ylabel('X2')
pp.show()
#grid off;

print " "
raw_input('Press any key to end')
pp.close(fh1)
pp.close(fh2)
