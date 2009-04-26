#!/usr/bin/env python
from netlab import *
import numpy as np
import matplotlib.pyplot as pp
from enthought.mayavi import mlab

#"""DEMGMM1 Demonstrate density modelling with a Gaussian mixture model.
#
#	Description
#	The problem consists of modelling data generated by a mixture of
#	three Gaussians in 2 dimensions.  The priors are 0.3, 0.5 and 0.2;
#	the centres are (2, 3.5), (0, 0) and (0,2); the variances are 0.2,
#	0.5 and 1.0. The first figure contains a  scatter plot of the data.
#
#	A Gaussian mixture model with three components is trained using EM.
#	The parameter vector is printed before training and after training.
#	The user should press any key to continue at these points.  The
#	parameter vector consists of priors (the column), centres (given as
#	(x, y) pairs as the next two columns), and variances (the last
#	column).
#
#	The second figure is a 3 dimensional view of the density function,
#	while the third shows the 1-standard deviation circles for the three
#	components of the mixture model.
#
#	See also
#	GMM, GMMINIT, GMMEM, GMMPROB, GMMUNPAK
#

#	Copyright (c) Ian T Nabney (1996-2001)
#       and Neil D. Lawrence (2009) (translation to python)"""

# Generate the data
# Fix seeds for reproducible results
np.random.seed(42)

ndata = 500
data, datac, datap, datasd = dem2ddat(ndata)

clc()
print "This demonstration illustrates the use of a Gaussian mixture model"
print "to approximate the unconditional probability density of data in"
print "a two-dimensional space.  We begin by generating the data from"
print "a mixture of three Gaussians and plotting it."
print " "
raw_input('Press return to continue')

fh1 = pp.figure()
pp.axes(axisbg='w', frameon=True)
pp.plot(data[:, 0], data[:, 1], 'o')
# Set up mixture model
ncentres = 3
input_dim = 2
mix = gmm(input_dim, ncentres, 'spherical')

options = foptions()
options[13] = 5	# Just use 5 iterations of k-means in initialisation
# Initialise the model parameters from the data
mix.init(data, options)

clc()
print "The data is drawn from a mixture with parameters"
print "    Priors        Centres         Variances"
print datap.T, datac, (datasd.T**2)
print " "
print "The mixture model has three components and spherical covariance"
print "matrices.  The model parameters after initialisation using the"
print "k-means algorithm are as follows"
# Print out model
print "    Priors        Centres            Variances"
print np.c_[mix.priors, mix.centres, mix.covars]
raw_input('Press return to continue')

# Set up vector of options for EM trainer
options = np.zeros(18)
options[0]  = 1		# Prints out error values.
options[13] = 10		# Max. Number of iterations.

print "We now train the model using the EM algorithm for 10 iterations"
print " "
raw_input('Press return to continue')
errlog = mix.em(data, options, returnFlog=True)

# Print out model
print " "
print "The trained model has parameters "
print "    Priors        Centres             Variances"
print np.c_[mix.priors, mix.centres, mix.covars]
print "Note the close correspondence between these parameters and those"
print "of the distribution used to generate the data, which are repeated here."
print "    Priors        Centres         Variances"
print np.c_[datap, datac, (datasd**2)]
print " "
raw_input('Press return to continue')

clc()
print "We now plot the density given by the mixture model as a surface plot"
print " "
raw_input('Press return to continue')

# Plot the result
x0=-4.0
x1=5.0
step=0.2
y0=x0
y1=x1

x = np.r_[x0:x1:step]
y = np.r_[y0:y1:step]
# 					
# Generate the grid
# 
X, Y = np.mgrid[x0:x1:step,y0:y1:step]

grid = np.c_[X.flatten(), Y.flatten()]
Z = mix.prob(grid)
Z = np.reshape(Z, (len(x), len(y))).T
#fig = mlab.figure()
#s = mlab.surf(x, y, Z)
#fig.add(s)
#hold on
#title('Surface plot of probability density')
#hold off

clc()
print "The final plot shows the centres and widths, given by one standard"
print "deviation, of the three components of the mixture model."
print " "
pp.show()
raw_input('Press return to continue.')
pp.ioff()

# Try to calculate a sensible position for the second figure, below the first
# fig1_pos = get(fh1, 'Position')
# fig2_pos = fig1_pos;
# fig2_pos(2) = fig2_pos(2) - fig1_pos(4);
fh2 = pp.figure()
# set(fh2, 'Position', fig2_pos)

hp1 = pp.plot(data[:, 0], data[:, 1], 'bo')
pp.axis('equal')
pp.hold(True)
hp2 = pp.plot(mix.centres[:, 0], mix.centres[:,1], 'g+')
pp.setp(hp2, markersize=10)
pp.setp(hp2, linewidth=3)
pp.title('Plot of data and mixture centres')
angles = np.r_[0:2*np.pi:np.pi/30]
for i in range( mix.ncentres):
  x_circle = mix.centres[i,0] + np.sqrt(mix.covars[i])*np.cos(angles)
  y_circle = mix.centres[i,1] + np.sqrt(mix.covars[i])*np.sin(angles)
  pp.plot(x_circle, y_circle, 'r')

pp.show()
pp.hold(False)
print "Note how the data cluster positions and widths are captured by"
print "the mixture model."
print " "
raw_input('Press return to end.')

pp.close(fh1)
pp.close(fh2)
