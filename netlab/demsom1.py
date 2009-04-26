#!/usr/bin/env python
from netlab import *
import numpy as np
import matplotlib.pyplot as pp
from matplotlib.font_manager import fontManager, FontProperties
import copy

"""#DEMSOM1 Demonstrate SOM for visualisation.
#
#	Description
#	 This script demonstrates the use of a SOM with  a two-dimensional
#	grid to map onto data in  two-dimensional space.  Both on-line and
#	batch training algorithms are shown.
#
#	See also
#	SOM, SOMPAK, SOMTRAIN
#

#	Copyright (c) Ian T Nabney (1996-2001)
#        and Neil D. Lawrence (2009) (translation to python)"""


np.random.seed(42)
nin = 2 
ndata = 300
# Give data an offset so that network has something to learn.
x = np.random.rand(ndata, nin) + np.array([1.5, 1.5])

clc()
print "This demonstration of the SOM, or Kohonen network, shows how the"
print "network units after training lie in regions of high data density."
print "First we show the data, which is generated uniformly from a square."
print "Red crosses denote the data and black dots are the initial locations"
print "of the SOM units."
print " "
raw_input('Press return to continue.')
net = som(nin, [8, 7])
c1 = net.pak()
h1 = pp.figure()
pp.show()
pp.plot(x[:, 0], x[:, 1], 'r+', label ='Data')
pp.hold(True)
pp.plot(c1[:,0], c1[:, 1], 'k.', label='Initial Weights')
options = foptions()

# Ordering phase
options[0] = 1
options[13] = 100
#options(14) = 5 # Just for testing
options[17] = 0.9  # Initial learning rate
options[15] = 0.05 # Final learning rate
options[16] = 8    # Initial neighbourhood size
options[14] = 1    # Final neighbourhood size

print "The SOM network is trained in two phases using an on-line algorithm."
print "Initially the neighbourhood is set to 8 and is then reduced"
print "linearly to 1 over the first 50 iterations."
print "Each iteration consists of a pass through the complete"
print "dataset, while the weights are adjusted after each pattern."
print "The learning rate is reduced linearly from 0.9 to 0.05."
print "This ordering phase puts the units in a rough grid shape."
print "Blue circles denote the units at the end of this phase."
print " "
raw_input('Press return to continue.')
net2 = copy.deepcopy(net)
net2.train(options, x)
c2 = net2.pak()
pp.plot(c2[:, 0], c2[:, 1], 'bo', label='Weights after ordering')

# Convergence phase
options[0] = 1
options[13] = 400
options[17] = 0.05
options[15] = 0.01
options[16] = 0
options[14] = 0

print "The second, convergence, phase of learning just updates the winning node."
print "The learning rate is reduced from 0.05 to 0.01 over 400 iterations."
print "Note how the error value does not decrease monotonically; it is"
print "difficult to decide when training is complete in a principled way."
print "The units are plotted as green hexagons."
print " "
raw_input('Press return to continue.')
net3 = copy.deepcopy(net2)
net3.train(options, x)
c3 = net3.pak()
pp.plot(c3[:, 0], c3[:, 1], 'gh', label='Weights after convergence')

# Now try batch training
options[0] = 1
options[5] = 1
options[13] = 50
options[16] = 3
options[14] = 0
print "An alternative approach to the on-line algorithm is a batch update"
print "rule.  Each unit is updated to be the average weights"
print "in a neighbourhood (which reduces from 3 to 0) over 50 iterations."
print "Note how the error is even more unstable at first, though eventually"
print "it does converge."
print "The final units are shown as black triangles."
print " "
raw_input('Press return to continue.')
#pp.ioff()
net4 = copy.deepcopy(net)
net4.train(options, x)
c4 = net4.pak()
pp.plot(c4[:, 0], c4[:, 1], 'k^', label='Batch weights')
font=FontProperties(size='small');
pp.legend(loc=2,numpoints=1,prop=font)

print " "
raw_input('Press return to end.')
print " "

pp.close(h1)
