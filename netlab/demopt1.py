#!/usr/bin/env python
from netlab import *
import numpy as np
import matplotlib.pyplot as pp

#function demopt1(xinit)
"""#DEMOPT1 Demonstrate different optimisers on Rosenbrock's function.
#
#	Description
#	The four general optimisers (quasi-Newton, conjugate gradients,
#	scaled conjugate gradients, and gradient descent) are applied to the
#	minimisation of Rosenbrock's well known `banana' function. Each
#	optimiser is run for at most 100 cycles, and a stopping criterion of
#	1.0e-4 is used for both position and function value. At the end, the
#	trajectory of each algorithm is shown on a contour plot of the
#	function.
#
#	DEMOPT1(XINIT) allows the user to specify a row vector with two
#	columns as the starting point.  The default is the point [-1 1]. Note
#	that the contour plot has an x range of [-1.5, 1.5] and a y range of
#	[-0.5, 2.1], so it is best to choose a starting point in the same
#	region.
#
#	See also
#	CONJGRAD, GRADDESC, QUASINEW, SCG, ROSEN, ROSEGRAD
#

#	Copyright (c) Ian T Nabney (1996-2001)
#        and Neil D. Lawrence (2009) (translation to python)"""


# try:

# Initialise start point for search
#if nargin < 1 | size(xinit) ~= [1 2]
#  xinit = [-1 1]	# Traditional start point
#end

flops_works = False

xinit = np.asarray([-1, 1])

# Set up options
options = foptions()	# Standard options
options[0] = -1 	# Turn off printing completely
options[2] = 1e-8 	# Tolerance in value of function
options[13] = 100  	# Max. 100 iterations of algorithm

clc()
print "This demonstration compares the performance of four generic"
print "optimisation routines when finding the minimum of Rosenbrock''s"
print "function y = 100*(x2-x1^2)^2 + (1-x1)^2."
print " "
print "The global minimum of this function is at [1 1]."
print "Each algorithm starts at the point [", xinit[0], ", ", xinit[1], "]."
print " "
raw_input("Press return to continue.")

# Generate a contour plot of the function
a = np.r_[-1.5:1.5:.02]
b = np.r_[-0.5:2.1:.02]

# Two alternative approaches for the next bit
if False:
    B, A = np.mgrid[-0.5:2.1:.02,-1.5:1.5:.02]
    Z = rosen(np.c_[A.flatten(1), B.flatten(1)])
    Z = np.reshape(Z, (len(b), len(a)), order='F')
    pdb.set_trace()
    l = np.r_[-1:6]
    v = pow(2, l)
    fh1 = pp.figure()
    pp.contour(a, b, Z, v)
else:
    A, B = np.mgrid[-1.5:1.5:.02,-0.5:2.1:.02]
    Z = rosen(np.c_[A.flatten(), B.flatten()])
    Z = np.reshape(Z, (len(a), len(b))).T
    pdb.set_trace()
    l = np.r_[-1:6]
    v = pow(2, l)
    fh1 = pp.figure()
    pp.contour(a, b, Z, v)

pp.title("Contour plot of Rosenbrock's function")
pp.hold(True)
pp.show()

clc()
print "We now use quasi-Newton, conjugate gradient, scaled conjugate"
print "gradient, and gradient descent with line search algorithms"
print "to find a local minimum of this function.  Each algorithm is stopped"
print "when 100 cycles have elapsed, or if the change in function value"
print "is less than 1.0e-8 or the change in the input vector is less than"
print "1.0e-4 in magnitude."
print " "
raw_input("Press return to continue.")

clc()
x = xinit
flops(0)
x, errlog, pointlog = quasinew(rosen, x, options, rosegrad, 
                                returnFlog=True, 
                                returnPoint = True)[:3]
print "For quasi-Newton method:"
print "Final point is (", x[0], ", ", x[1], "), value is ", options[7]
print "Number of function evaluations is ", options[9]
print "Number of gradient evaluations is ", options[10]
if flops_works:
    opt_flops = flops()
    print "Number of floating point operations is ", opt_flops

print "Number of cycles is", len(pointlog) - 1
print " "

x = xinit
flops(0)
x, errlog2, pointlog2 = conjgrad(rosen, x, options, rosegrad, 
                                returnFlog=True, 
                                returnPoint = True)[:3]
print "For conjugate gradient method:"
print "Final point is (", x[0], ', ', x[1], '), value is ', options[7]
print "Number of function evaluations is ", options[9]
print "Number of gradient evaluations is ", options[10]
if flops_works:
    opt_flops = flops()
    print 'Number of floating point operations is ', opt_flops

print "Number of cycles is ", len(pointlog2) - 1
print " "

x = xinit
flops(0)
x, errlog3, pointlog3 = scg(rosen, x, options, rosegrad,
                            returnFlog=True,
                            returnPoint=True)[:3]
print "For scaled conjugate gradient method:"
print "Final point is (", x[0], ", ", x[1], "), value is ", options[7]
print "Number of function evaluations is ", options[9]
print "Number of gradient evaluations is ", options[10]
if flops_works:
    opt_flops = flops()
    print "Number of floating point operations is ", opt_flops

print "Number of cycles is ", len(pointlog3) - 1
print " "

x = xinit
options[6] = 1 # Line minimisation used

flops(0)
x, errlog4, pointlog4 = graddesc(rosen, x, options, rosegrad, 
                                 returnFlog=True, 
                                 returnPoint=True)[:3]
print "For gradient descent method:"
print "Final point is (", x[0], ", ", x[1], ") value is ", options[7]
print "Number of function evaluations is ", options[9]
print "Number of gradient evaluations is ", options[10]
if flops_works:
    opt_flops = flops()
    print "Number of floating point operations is ", opt_flops

print "Number of cycles is ", len(pointlog4) - 1
print " "
print "Note that gradient descent does not reach a local minimum in"
print "100 cycles."
print " "
print "On this problem, where the function is cheap to evaluate, the"
print "computational effort is dominated by the algorithm overhead."
print "However on more complex optimisation problems (such as those"
print "involving neural networks), computational effort is dominated by"
print "the number of function and gradient evaluations.  Counting these,"
print "we can rank the algorithms: quasi-Newton (the best), conjugate"
print "gradient, scaled conjugate gradient, gradient descent (the worst)"
print " "
raw_input("Press return to continue.")
clc()
print "We now plot the trajectory of search points for each algorithm"
print "superimposed on the contour plot."
print " "
raw_input("Press return to continue.")
pp.plot(np.asarray(pointlog4)[:,0], np.asarray(pointlog4)[:,1], 'bd', markersize = 6, label = 'Gradient Descent')
pp.plot(np.asarray(pointlog3)[:,0], np.asarray(pointlog3)[:,1], 'mx', markersize = 6, linewidth = 2, label = 'Scaled Conjugate Gradients')
pp.plot(np.asarray(pointlog)[:,0], np.asarray(pointlog)[:,1], 'k.', markersize = 18, label = 'Quasi Newton')
pp.plot(np.asarray(pointlog2)[:,0], np.asarray(pointlog2)[:,1], 'g+', markersize = 6, linewidth = 2, label = 'Conjugate Gradients')
lh = pp.legend()
pp.show()
pp.hold(False)

clc()
raw_input("Press return to end.")
pp.close(fh1)
#clear all

# except:
#     import pdb, sys
#     e, m, tb = sys.exc_info()
#     pdb.post_mortem(tb)
