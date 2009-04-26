#!/usr/bin/env python
from netlab import *
import numpy as np
import matplotlib.pyplot as pp

#"""DEMMLP2 Demonstrate simple classification using a multi-layer perceptron
#
#	Description
#	The problem consists of input data in two dimensions drawn from a
#	mixture of three Gaussians: two of which are assigned to a single
#	class.  An MLP with logistic outputs trained with a quasi-Newton
#	optimisation algorithm is compared with the optimal Bayesian decision
#	rule.
#
#	See also
#	MLP, MLPFWD, NETERR, QUASINEW
#

#	Copyright (c) Ian T Nabney (1996-2001)
#        and Neil D. Lawrence (2009) (translation to python)"""


# Set up some figure parameters
AxisShift = 0.05
ClassSymbol1 = 'r.'
ClassSymbol2 = 'y.'
PointSize = 12
titleSize = 10

# Fix the seeds
np.random.seed(423)

clc()
print "This demonstration shows how an MLP with logistic outputs and"
print "and cross entropy error function can be trained to model the"
print "posterior class probabilities in a classification problem."
print "The results are compared with the optimal Bayes rule classifier,"
print "which can be computed exactly as we know the form of the generating"
print "distribution."
print " "
raw_input("Press return to continue.")

fh1 = pp.figure()
fh1.suptitle('True Data Distribution')
#whitebg(fh1, 'k')

# 
# Generate the data
# 
n=200

# Set up mixture model: 2d data with three centres
# Class 1 is first centre, class 2 from the other two
mix = gmm(2, 3, 'full')
mix.priors = np.array([0.5, 0.25, 0.25])
mix.centres = np.array([[0, -0.1], [1, 1], [1, -1]])
mix.covars[:,:,0] = np.array([[0.625, -0.2165], [-0.2165, 0.875]])
mix.covars[:,:,1] = np.array([[0.2241, -0.1368], [-0.1368, 0.9759]])
mix.covars[:,:,2] = np.array([[0.2375, 0.1516], [0.1516, 0.4125]])

[data, label] = mix.samp(n, returnLabels=True)

# 
# Calculate some useful axis limits
# 
x0 = data[:,0].min()
x1 = data[:,0].max()
y0 = data[:,1].min()
y1 = data[:,1].max()
dx = x1-x0
dy = y1-y0
expand = 5/100			# Add on 5 percent each way
x0 = x0 - dx*expand
x1 = x1 + dx*expand
y0 = y0 - dy*expand
y1 = y1 + dy*expand
resolution = 100
step = dx/resolution
xrange = np.r_[x0:x1:step]
yrange = np.r_[y0:y1:step]
# 					
# Generate the grid
# 
X, Y = np.mgrid[x0:x1:step,y0:y1:step]
# 
# Calculate the class conditional densities, the unconditional densities and
# the posterior probabilities
# 
px_j = mix.activ(np.c_[X.flatten(), Y.flatten()])
px = np.reshape(np.dot(px_j,(mix.priors).T), X.shape).T
post = mix.post(np.c_[X.flatten(), Y.flatten()])[0]
p1_x = np.reshape(post[:, 0], X.shape).T
p2_x = np.reshape(post[:, 1] + post[:, 2], X.shape).T

# 
# Generate some pretty pictures !!
# 
pp.hot()
pp.colorbar
pp.subplot(1,2,1)
pp.hold(True)
pp.plot(data[np.nonzero(label==0)[0],0],data[np.nonzero(label==0)[0],1],ClassSymbol1, markersize=PointSize)
pp.plot(data[np.nonzero(label>0)[0],0],data[np.nonzero(label>0)[0],1],ClassSymbol2, markersize=PointSize)
pp.contour(xrange,yrange,p1_x, [0.5, 0.5])
pp.show()

pp.axis([x0, x1, y0, y1])
#set(gca,'Box','On')
pp.title('The Sampled Data')
#rect=get(gca,'Position')
#rect(1)=rect(1)-AxisShift
#rect(3)=rect(3)+AxisShift
#set(gca,'Position',rect)
#hold off

pp.subplot(1,2,2)
pp.imshow(px, extent=[x0, x1, y0, y1], origin='lower')
pp.hold(True)
pp.contour(xrange,yrange,p1_x, [0.5, 0.5])
# set(hB,'LineWidth', 2)
pp.axis([x0, x1, y0, y1])
# set(gca,'YDir','normal')
pp.title(r'Probability Density $p(x)$')
pp.hold(False)

pp.show()
clc()
print "The first figure shows the data sampled from a mixture of three"
print "Gaussians, the first of which (whose centre is near the origin) is"
print "labelled red and the other two are labelled yellow.  The second plot"
print "shows the unconditional density of the data with the optimal Bayesian"
print "decision boundary superimposed."
print " "
raw_input("Press return to continue.")

fh2 = pp.figure()
fh2.canvas.set_window_title('Class-conditional Densities and Posterior Probabilities')
# whitebg(fh2, 'w')
# pp.setp(fh2, facecolor=[1, 1, 1])

pp.subplot(2,2,1)
p1=np.reshape(px_j[:,0],X.shape).T
pp.imshow(p1, extent=[x0, x1, y0, y1], origin='lower')
pp.hot()
pp.colorbar()
# set(gca,'YDir','normal')
pp.hold(True)
pp.plot(mix.centres[:,0],mix.centres[:,1],'b+',markersize=8,linewidth=2)
pp.title(r'Density $p(x|\mathrm{red})$')
pp.axis('image')
pp.hold(False)

pp.subplot(2,2,2)
p2=np.reshape(px_j[:,1]+px_j[:,2], X.shape).T
pp.imshow(p2, extent=[x0, x1, y0, y1], origin='lower')
pp.colorbar()
#set(gca,'YDir','normal')
pp.hold(True)
pp.plot(mix.centres[:,0],mix.centres[:,1],'b+', markersize=8, linewidth=2)
pp.title(r'Density $p(x|\mathrm{yellow})$')
pp.axis('image')
pp.hold(False)

pp.subplot(2,2,3)
pp.imshow(p1_x, extent=[x0, x1, y0, y1], origin='lower')
#set(gca,'YDir','normal')
pp.colorbar()
pp.title(r'Posterior Probability $p(\mathrm{red}|x)$')
pp.hold(True)
pp.plot(mix.centres[:,0],mix.centres[:,1],'b+', markersize=8, linewidth=2)
pp.axis('image')
pp.hold(False)

pp.subplot(2,2,4)
pp.imshow(p2_x, extent=[x0, x1, y0, y1], origin='lower')
#set(gca,'YDir','normal')
pp.colorbar()
pp.title(r'Posterior Probability $p(\mathrm{yellow}|x)$')
pp.hold(True)
pp.plot(mix.centres[:,0],mix.centres[:,1],'b+',markersize=8,linewidth=2)
pp.axis('image')
pp.hold(False)

# Now set up and train the MLP
nhidden=6
nout=1
alpha = 0.2	# Weight decay
ncycles = 60	# Number of training cycles. 
# Set up MLP network
net = mlp(2, nhidden, nout, 'logistic', alpha)
options = np.zeros(18)
options[0] = 1                 # Print out error values
options[8] = 1
options[13] = ncycles

mlpstring = 'We now set up an MLP with ' + str(nhidden) + \
    ' hidden units, logistic output and cross'
trainstring = 'entropy error function, and train it for ' \
    + str(ncycles) + ' cycles using the'
wdstring = 'quasi-Newton optimisation algorithm with weight decay of ' \
    + str(alpha) + '.'

# Force out the figure before training the MLP
pp.show()
print " "
print "The second figure shows the class conditional densities and posterior"
print "probabilities for each class. The blue crosses mark the centres of"
print "the three Gaussians."
print " "
print mlpstring
print trainstring
print wdstring
print " "
raw_input("Press return to continue.")

# Convert targets to 0-1 encoding
target = np.zeros((n, 1))
target[np.nonzero(label==1)[0]] = 1.0

# Train using quasi-Newton.
net = netopt(net, options, data, target, quasinew)
y = net.fwd(data)[0]
yg = net.fwd(np.c_[X.flatten(), Y.flatten()])[0]
yg = np.reshape(yg, X.shape).T

fh3 = pp.figure()
fh3.canvas.set_window_title('Network Output')
# whitebg(fh3, 'k')
pp.subplot(1, 2, 1)
pp.hold(True)
pp.plot(data[np.nonzero(label==0)[0],0],data[np.nonzero(label==0)[0],1],'r.', markersize=PointSize)
pp.plot(data[np.nonzero(label>0)[0],0],data[np.nonzero(label>0)[0],1],'y.', markersize=PointSize)
# Bayesian decision boundary
# [cB, hB] = contour(xrange,yrange,p1_x,[0.5 0.5],'b-')
pp.contour(xrange,yrange,p1_x,[0.5, 0.5])
# [cN, hN] = contour(xrange,yrange,yg,[0.5 0.5],'r-')
pp.contour(xrange,yrange,yg,[0.5, 0.5])
# set(hB, 'LineWidth', 2)
# set(hN, 'LineWidth', 2)
# Chandles = [hB(1) hN(1)]
# legend(Chandles, 'Bayes', 'Network', 3)

pp.axis([x0, x1, y0, y1])
# set(gca,'Box','on','XTick',[],'YTick',[])

pp.title('Training Data',fontsize=titleSize)
pp.hold(False)

pp.subplot(1, 2, 2)
pp.imshow(yg, extent=[x0, x1, y0, y1], origin='lower')
pp.hot()
pp.colorbar()
# axis(axis)
# set(gca,'YDir','normal','XTick',[],'YTick',[])
pp.title('Network Output',fontsize=titleSize)

clc()
print "This figure shows the training data with the decision boundary"
print "produced by the trained network and the network''s prediction of"
print "the posterior probability of the red class."
print " "
raw_input("Press return to continue.")

# # 
# # Now generate and classify a test data set
# # 
testdata, testlabel = mix.samp(n, returnLabels=True)
testlab = np.c_[testlabel==0, testlabel>0]

# # This is the Bayesian classification
tpx_j = mix.post(testdata)[0]
Bpost = np.c_[tpx_j[:,0], tpx_j[:,1]+tpx_j[:,2]]
Bcon, Brate=confmat(Bpost, np.c_[testlabel==0, testlabel>0])

# Compute network classification
yt = net.fwd(testdata)[0]
# Convert single output to posteriors for both classes
testpost = np.c_[yt, 1-yt]
#C, trate=confmat(testpost,np.c_[testlabel==0, testlabel>0])
C, trate=confmat(yt,testlabel-1)

fh4 = pp.figure()
fh4.canvas.set_window_title('Decision Boundaries')
# whitebg(fh4, 'k')
pp.hold(True)
pp.plot(testdata[np.nonzero(testlabel==0)[0],0],
            testdata[np.nonzero(testlabel==0)[0],1],
            ClassSymbol1, markersize=PointSize)
pp.plot(testdata[np.nonzero(testlabel>0)[0],0],
            testdata[np.nonzero(testlabel>0)[0],1],
            ClassSymbol2, markersize=PointSize)
# Bayesian decision boundary
# [cB, hB] = contour(xrange,yrange,p1_x,[0.5 0.5],'b-')
pp.contour(xrange,yrange,p1_x,[0.5, 0.5])
# set(hB, 'LineWidth', 2)
# Network decision boundary
# [cN, hN] = contour(xrange,yrange,yg,[0.5 0.5],'r-')
pp.contour(xrange,yrange,yg,[0.5, 0.5])
# set(hN, 'LineWidth', 2)
# Chandles = [hB(1) hN(1)]
# legend(Chandles, 'Bayes decision boundary', ...
#   'Network decision boundary', -1)
pp.axis([x0, x1, y0, y1])
pp.title('Test Data')
# set(gca,'Box','On','Xtick',[],'YTick',[])

clc()
print "This figure shows the test data with the decision boundary"
print "produced by the trained network and the optimal Bayes rule."
print " "
raw_input("Press return to continue.")

fh5 = pp.figure()
fh5.canvas.set_window_title('Test Set Performance')
# whitebg(fh5, 'w')
# Bayes rule performance
pp.subplot(1,2,1)
plotmat(Bcon,'b','k',12)
# set(gca,'XTick',[0.5 1.5])
# set(gca,'YTick',[0.5 1.5])
# grid('off')
# set(gca,'XTickLabel',['Red   ' ; 'Yellow'])
# set(gca,'YTickLabel',['Yellow' ; 'Red   '])
pp.ylabel('True')
pp.xlabel('Predicted')
pp.title('Bayes Confusion Matrix (' + str(Brate[0]) + '%)')

# Network performance
pp.subplot(1,2, 2)
plotmat(C,'b','k',12)
# set(gca,'XTick',[0.5 1.5])
# set(gca,'YTick',[0.5 1.5])
# grid('off')
# set(gca,'XTickLabel',['Red   ' ; 'Yellow'])
# set(gca,'YTickLabel',['Yellow' ; 'Red   '])
pp.ylabel('True')
pp.xlabel('Predicted')
pp.title('Network Confusion Matrix (' + str(trate[0]) + '%)')

print "The final figure shows the confusion matrices for the"
print "two rules on the test set."
print " "
raw_input("Press return to exit.")

# whitebg(fh1, 'w')
# whitebg(fh2, 'w')
# whitebg(fh3, 'w')
# whitebg(fh4, 'w')
# whitebg(fh5, 'w')
pp.close(fh1) 
pp.close(fh2) 
pp.close(fh3)
pp.close(fh4) 
pp.close(fh5)
# clear all;

