#!/usr/bin/env python
from netlab import *
import numpy as np
import matplotlib.pyplot as pp

# """DEMKMEAN Demonstrate simple clustering model trained with K-means.
# 	Description
# 	The problem consists of data in a two-dimensional space.  The data is
# 	drawn from three spherical Gaussian distributions with priors 0.3,
# 	0.5 and 0.2; centres (2, 3.5), (0, 0) and (0,2); and standard
# 	deviations 0.2, 0.5 and 1.0. The first figure contains a scatter plot
# 	of the data.  The data is the same as in DEMGMM1.

# 	A cluster model with three components is trained using the batch K-
# 	means algorithm. The matrix of centres is printed after training. The
# 	second figure shows the data labelled with a colour derived from the
#         corresponding  cluster

# 	See also
# 	DEM2DDAT, DEMGMM1, KNN1, KMEANS

# 	Copyright (c) Ian T Nabney (1996-2001)
#        and Neil D. Lawrence (2009) (translation to python)"""

try:
  # Generate the data, fixing seeds for reproducible results
  ndata = 250
  np.random.seed(42)

  data, c, prior, sd = dem2ddat(ndata)

  # Randomise data order
  data = data[np.random.permutation(ndata), :]

  clc()
  print "This demonstration illustrates the use of a cluster model to"
  print "find centres that reflect the distribution of data points."
  print "We begin by generating the data from a mixture of three Gaussians"
  print "in two-dimensional space and plotting it."
  print " "
  raw_input("Press return to continue.")

  fh1 = pp.figure()
  pp.plot(np.asarray(data[:, 0]), np.asarray(data[:, 1]), 'o')
  pp.gca().set_frame_on(True)
  pp.title('Data')
  fh1.show()
  # Set up cluster model
  ncentres = 3;
  centres = np.asmatrix(np.zeros((ncentres, 2)))

  # Set up vector of options for kmeans trainer
  options = foptions()
  options[0]  = 1		# Prints out error values.
  options[4] = 1
  options[13] = 10		# Number of iterations.

  clc()
  print "The model is chosen to have three centres, which are initialised"
  print "at randomly selected data points.  We now train the model using"
  print "the batch K-means algorithm with a maximum of 10 iterations and"
  print "stopping tolerance of 1e-4."
  print " "
  raw_input("Press return to continue.")

  # Train the centres from the data
  centres, post, errlog = kmeans(centres, data, options)

  # Print out model
  print " "
  print "Note that training has terminated before 10 iterations as there"
  print "has been no change in the centres or error function."
  print " "
  print "The trained model has centres:"
  print centres
  raw_input("Press return to continue.")

  clc()
  print "We now plot each data point coloured according to its classification"
  print "given by the nearest cluster centre.  The cluster centres are denoted"
  print "by black crosses."

  # 					Plot the result
  fh2 = pp.figure()

  fh2.hold(True)
  colours = ['b.', 'r.', 'g.']

  tempi, tempj = post.nonzero()
  fh2.hold(True)
  hp = list()
  for i in range(3):
    # Select data points closest to ith centre
    thisX = np.asarray(data[tempi[(tempj == i).nonzero()], 0])
    thisY = np.asarray(data[tempi[(tempj == i).nonzero()], 1])
    hp.append(pp.plot(thisX, thisY, colours[i]))
    pp.setp(hp[i], 'MarkerSize', 12)

  pp.gca().set_frame_on(True)
  pp.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
  fh2.hold(True)
  ph = pp.plot(np.asarray(centres[:, 0]), np.asarray(centres[:,1]), 'k+')
  pp.setp(ph, 'LineWidth', 3, 'MarkerSize', 8)
  pp.title('Centres and data labels')
  fh2.hold(False)
  fh2.show()
  print " "
  raw_input("Press return to end.")

  pp.close(fh1)
  pp.close(fh2)

except:
  import pdb, sys
  e, m, tb = sys.exc_info()
  pdb.post_mortem(tb)
  
