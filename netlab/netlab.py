
""" 
From Roland Memisevic:

http://www.learning.cs.toronto.edu/~rfm/m2py/

ipython -pylab




Important things:

reload moduleName

in ipython, use object? to get lots of info.

Debugging:

pdb on

or

import pdb

pdb.set_trace()

in ipython just write pdb

No switch statement in python!

Annoying things in python. The difference between matrix and array behavior. Try 
a = np.asmatrix(randn(100, 1))
b = a.T*a
b.shape

a = np.asarray(randn(100, 1))
b = a.T*a
b.shape

# explanation ... * changes behavior between the two. First it is matrix multiply. For array it isn't.

We can be more explicit. dot() gives matrix multiplaction for arrays.

a = np.asmatrix(randn(100, 1))
b = dot(a.T, a)
b.shape

a = np.asarray(randn(100, 1))
b = dot(a.T, a)
b.shape

What happens with a.T*a in the array case? We can force array behavior with multiply.

a = np.asmatrix(randn(100, 1))
b = multiply(a.T, a)
b.shape

a = np.asarray(randn(100, 1))
b = multiply(a.T, a)
b.shape

This means .* in MATLAB, but it has the added useful/confusing behavior that it automatically tiles to form the multiplication.

Consider this MATLAB construct.

a = exp(randn(10, 400))
suma = sum(a, 1)
b = a./repmat(suma, 10, 1)
size(b)
Note the repmat in there. Instead in python this can be done with:

a = np.exp(randn(10, 400))
b = a/a.sum(0)
b.shape

Here, sum is summing over the first dimension (python indexes start from 0 in python) and automatically doing the repmat (tiling) for us! Neat eh? This also works with matrices, 

a = np.asmatrix(np.exp(randn(10, 400)))
b = a/a.sum(0)
b.shape

Of course we should use things in design matrix format, so we have

a = np.asmatrix(np.exp(randn(400, 10)))
b = a/a.sum(1)
b.shape

And finally, let's just check that works with arrays ...

a = np.asarray(np.exp(randn(400, 10)))
b = a/a.sum(1)
b.shape

It doesn't work ... the problem is that the result of the sum in array is a one dimensional array and you can't do the automatic repmat!!

These behaviours are nasty because your code will work/fail simply dependent on whether someone has fed you an array or a matrix.

The repmat automatic tiling can also be a pain ... as it happens automaticaly, and can hide dimension errors.




Other Gotchas
=============

 a = [1 2 3 4; 5 6 7 8];
 reshape(a, 4, 2)


 a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
 a.reshape(4, 2)
 print a

The problem is because python follows C. Row-major order is used in C and Python; column-major order is used in Fortran and Matlab.

Fix is to use

a.reshape(4, 2, 'F')

Same issues apply to "flatten".

a = [1 2; 3 4]
a(:)'


np.array([[1, 2], [3, 4]]).flatten()

Instead you have to use

np.array([[1, 2], [3, 4]]).flatten(1).T

Again the issue of arrays changing dimension rears its head here.

a = ones(1, 10)
b = ones(10, 10)
c = [b(:)' a]

a = np.ones((1, 10))
b = np.ones((10, 10))
c= r_[b.flatten(1).T, a]


Zeros and randn different behavior

# Python
np.zeros(1, 10)
np.zeros((1, 10))
np.zeros(10)

np.random.randn((1, 10))
np.random.randn(1, 10)
np.random.randn(10)

% MATLAB
randn(10)


Indexing
--------

Similar to matlab, but ranges in python stop before the highest number:

a = [1 2 3 4];
a(1:3)

a = np.array([1, 2, 3, 4])
print a[0:3]

Also beware that the step parameter comes at the end in numpy.

a = 1:10:200

a = r_[1:200:10]

The end value in matlab is replaced with -1. Any -ve number is considered to be indexing from the end, i.e. -2 is end-1, -3 is end-2 etc. Although it will stop before that end number ... need to use [0:] to go to end ...

To reverse the indexing of an array 

a(end:-1:1)

becomes

a[::-1]

Beware the difference between


a = [1 2; 3 4]
a(1) = 0.0
a

and

a = np.array([[1, 2], [3, 4]])
a[0] = 0.0
print a

This can catch you out if the array with

a = np.random.randn(1, 40)
p a[0]

It is particularly confusing as for one dimensional arrays it works fine ... but the problems start if you start by saying a = zeros(18) vs a = zeros(1, 18)

np.asarray(randn(100, 1)).sum(0)

np.asarray(randn(100, 1)).sum()


Editing
=======

After editing modules you need to reload the module.

Plotting
========


plot(plotvals, y, 'k-', 'linewidth', 2) becomes
pp.plot(plotvals.T, y.T, 'k-', linewidth=2)

Matplotlib seems to accept only arrays not matrices!

cov
===

The cov command assumes things are the wrong way around.

cov(randn(100, 2))

np.cov(np.randn(100, 2))

use np.cov(np.randn(100, 2), rowvar=0)


Rank in MATLAB and Python
=========================
In MATLAB rank estimates the rank of a matrix through svd

rank([0, 1, 2, 3; 0, 2, 4, 6; 3, 8, 2, 3; 4, 2, 1, 5])

equivalent to 

A = [0, 1, 2, 3; 0, 2, 4, 6; 3, 8, 2, 3; 4, 2, 1, 5]
s = svd(A)
tol = max(diag(A))*eps(max(s))
r = sum(s > tol)


in python it gives the dimension

np.rank([[0, 1, 2, 3],[0, 2, 4, 6],[3, 8, 2, 3],[ 4, 2, 1, 5]])

LAMBDA
======

lambda is a keyword in python


Tile and Repmat
===============
a = [1; 2; 3; 4]
size(a)
repmat(a, [2, 3])
size(a)

a = np.array([[1], [2], [3], [4]])
a.shape
np.tile(a, (2, 3))

Mgrid
=====

Returns arguments in a different order from meshgrid.
[X, Y] = meshgrid(0:3, 0:4)
Y, X = mgrid[0:4,0:5]

Bizarre behaviour (bug?)
========================

a = [1; 2; 3; 4]
size(a)
b = repmat(a, [1, 1, 2])
size(b)

a = np.array([[1], [2], [3], [4]])
a.shape
b = np.tile(a, (1, 1, 2))
b.shape

A fix I found was to do

b = np.tile(a, (a.shape[0], a.shape[1], 2)) 

for this case.


Sum

sum(np.random.randn(100, 3), 1).shape
np.sum(np.random.randn(100, 3), 1).shape


"""
import pdb

import numpy as np
import numpy.linalg as la
import matplotlib.mlab as ml
import matplotlib.pyplot as pp

import types
import math


def confmat(Y,T):
    """CONFMAT Compute a confusion matrix.

    Description
    [C, RATE] = CONFMAT(Y, T) computes the confusion matrix C and
    classification performance RATE for the predictions mat{y} compared
    with the targets T.  The data is assumed to be in a 1-of-N encoding,
    unless there is just one column, when it is assumed to be a 2 class
    problem with a 0-1 encoding.  Each row of Y and T corresponds to a
    single example.
    
    In the confusion matrix, the rows represent the true classes and the
    columns the predicted classes.  The vector RATE has two entries: the
    percentage of correct classifications and the total number of correct
    classifications.
    
    See also
    CONFFIG, DEMTRAIN
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""

    n, c = Y.shape
    n2, c2 = T.shape

    if n != n2 or c != c2:
        raise Exception('Outputs and targets are different sizes')

    if c > 1:
        # Find the winning class assuming 1-of-N encoding
        Yclass = np.argmax(Y, axis=1)+1
        TL=np.dot(np.arange(1,c+1),T.T)

    else:
        # Assume two classes with 0-1 encoding
        c = 2
        class2 = np.nonzero(T > 0.5)[0]
        TL = np.ones(n)
        TL[class2] = 2
        class2 = np.nonzero(Y > 0.5)[0]
        Yclass = np.ones(n)
        Yclass[class2] = 2
    # Compute 
    pdb.set_trace()
    correct = (Yclass==TL)
    total = correct.sum()
    rate = np.array([total*100/n, total])

    C = np.zeros((c,c))
    for i in range(c):
        for j in range(c):
            C[i,j] = np.sum(np.multiply((Yclass==j+1),(TL==i+1)))
    return C,rate 

def dist2(x, c):
    """Calculates squared distance between two sets of points.
	Description
	D = DIST2(X, C) takes two matrices of vectors and calculates the
	squared Euclidean distance between them.  Both matrices must be of
	the same column dimension.  If X has M rows and N columns, and C has
	L rows and N columns, then the result has M rows and L columns.  The
	I, Jth entry is the  squared distance from the Ith row of X to the
	Jth row of C.

	See also
	GMMACTIV, KMEANS, RBFFWD

	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""
    
    (ndata, dimx) = x.shape
    (ncentres, dimc) = c.shape
    if not dimx == dimc:
        raise Exception('Data dimension does not match dimension of centres')
        
    n2 = np.tile(np.multiply(x,x).sum(1).reshape(-1, 1), (1, ncentres)) \
        + np.tile(np.multiply(c,c).sum(1).reshape(-1, 1), (1, ndata)).T \
        - 2*np.dot(x,c.T)

    # Rounding errors occasionally cause negative entries in n2
    if np.any(n2<0):
        n2[np.nonzero(n2<0)] = 0
    return n2


def pca(data, N=None):
    """PCA	Principal Components Analysis

	Description
	 PCCOEFF = PCA(DATA) computes the eigenvalues of the covariance
	matrix of the dataset DATA and returns them as PCCOEFF.  These
	coefficients give the variance of DATA along the corresponding
	principal components.

	PCCOEFF = PCA(DATA, N) returns the largest N eigenvalues.

	[PCCOEFF, PCVEC] = PCA(DATA) returns the principal components as well
	as the coefficients.  This is considerably more computationally
	demanding than just computing the eigenvalues.

	See also
	EIGDEC, GTMINIT, PPCA

	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

    #data = np.asmatrix(data)

    if N == None:
        N = data.shape[1]

    # Would be true if you are returning only eigenvectors.
    evals_only = False

    if not N == round(N) or N < 1 or N > data.shape[1]:
        raise Exception('Number of PCs must be integer, >0, < dim')

    # Find the sorted eigenvalues of the data covariance matrix
    if evals_only:
        PCcoeff = eigdec(np.cov(data, rowvar=0), N)
    else:
        PCcoeff, PCvec = eigdec(np.cov(data, rowvar=0), N)

    # Return real part of eigenvectors (for numerical reasons).
    return PCcoeff, np.real(PCvec)

def gauss(mu, covar, x):
    """GAUSS	Evaluate a Gaussian distribution.
    
	Description
        
	Y = GAUSS(MU, COVAR, X) evaluates a multi-variate Gaussian  density
	in D-dimensions at a set of points given by the rows of the matrix X.
	The Gaussian density has mean vector MU and covariance matrix COVAR.

	See also
	GSAMP, DEMGAUSS


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""
    
    n, d = x.shape

    j, k = covar.shape

    # Check that the covariance matrix is the correct dimension
    if ((j != d) or (k !=d)):
        raise Exception('Dimension of the covariance matrix and data should match');
   
    invcov = la.inv(covar)
    mu.reshape((1, -1))    # Ensure that mu is a row vector

    x = x - mu
    fact = np.sum(((np.dot(x,invcov))*x), 1)

    y = np.exp(-0.5*fact)

    y = y/np.sqrt((2*np.pi)**d*la.det(covar))
    return y

def gsamp(mu, covar, nsamp):
    """GSAMP	Sample from a Gaussian distribution.
    
    Description
    
    X = GSAMP(MU, COVAR, NSAMP) generates a sample of size NSAMP from a
    D-dimensional Gaussian distribution. The Gaussian density has mean
    vector MU and covariance matrix COVAR, and the matrix X has NSAMP
    rows in which each row represents a D-dimensional sample vector.
    
    See also
    GAUSS, DEMGAUSS
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""

    d = covar.shape[0]

    mu.reshape(1, d)   # Ensure that mu is a row vector

    eval, evec = la.eig(covar)

    deig=eval.reshape(d, 1)

    if np.any(np.iscomplex(deig)) or any(deig<0): 
        print 'Covariance Matrix is not OK, redefined to be positive definite'
        eval=abs(eval)

    coeffs = np.dot(np.random.randn(nsamp, d), np.diag(np.sqrt(eval)))

    x = mu + np.dot(coeffs, evec.T)
    return x

def plotmat(matrix, textcolour, gridcolour, fontsize):
    """PLOTMAT Display a matrix.

    Description
    PLOTMAT(MATRIX, TEXTCOLOUR, GRIDCOLOUR, FONTSIZE) displays the matrix
    MATRIX on the current figure.  The TEXTCOLOUR and GRIDCOLOUR
    arguments control the colours of the numbers and grid labels
    respectively and should follow the usual Matlab specification. The
    parameter FONTSIZE should be an integer.
    
    See also
    CONFFIG, DEMMLP2
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""
    
    m,n=matrix.shape
    for rowCnt in range(m):
        for colCnt in range(n):
            numberString=str(matrix[rowCnt,colCnt])
            pp.text(colCnt+.5, m-rowCnt-.5, numberString,
                    ha='center',
                    color=textcolour,
                    fontweight='bold',
                    fontsize=fontsize)

        pp.setp(pp.gca(),clip_box='on',
                visible='on',
                xlim=[0, n],
                xticklabels=[],
                xticks=np.arange(0,n),
                ylim=[0, m],
                yticklabels=[],
                yticks=np.arange(0,m))
    

def ppca(x, ppca_dim):
    """PPCA	Probabilistic Principal Components Analysis
    
    Description
    [VAR, U, LAMBDA] = PPCA(X, PPCA_DIM) computes the principal
    component subspace U of dimension PPCA_DIM using a centred covariance
    matrix X. The variable VAR contains the off-subspace variance (which
    is assumed to be spherical), while the vector LAMBDA contains the
    variances of each of the principal components.  This is computed
    using the eigenvalue and eigenvector  decomposition of X.
    
    See also
    EIGDEC, PCA
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""


    if ppca_dim != round(ppca_dim) or ppca_dim < 1 or ppca_dim > x.shape[1]:
        raise Exception('Number of PCs must be integer, >0, < dim')
    lambd = np.zeros(ppca_dim)
    ndata, data_dim = x.shape
    # Assumes that x is centred and responsibility weighted
    # covariance matrix
    l, Utemp = eigdec(x, data_dim)
    # Zero any negative eigenvalues (caused by rounding)
    l[np.nonzero(l<0)] = 0
    # Now compute the sigma squared values for all possible values
    # of q
    s2_temp = np.cumsum(l[::-1])/np.r_[1:data_dim+1]
    # If necessary, reduce the value of q so that var is at least
    # eps * largest eigenvalue
    q_temp = min(ppca_dim, data_dim - 1 - np.nonzero(s2_temp/l[0] > eps())[0].min())
    if q_temp != ppca_dim:
        wstringpart = 'Covariance matrix ill-conditioned: extracted'
        wstring = wstringpart + str(q_temp) + '/' + str(ppca_dim)
        print "Warning: ", wstring
    if q_temp == 0:
        # All the latent dimensions have disappeared, so we are
        # just left with the noise model
        vr = l[0]/data_dim
        lambd = vr*np.ones(ppca_dim)
    else:
        vr = l[q_temp:].mean()
    U = Utemp[:, 0:q_temp]
    lambd = l[0:q_temp]
    return vr, U, lambd



def eigdec(x, N):
    """EIGDEC	Sorted eigendecomposition

	Description
        EVALS, EVEC = EIGDEC(X, N) computes the largest N eigenvalues of the
	matrix X in descending order.  

	See also
	PCA, PPCA

	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

    #x = np.asmatrix(x)
    # Would be true if you are returning only eigenvectors, can't make
    # that decision in python
    evals_only = False
    
    if not N == round(N) or N < 1 or N > x.shape[1]:
        raise Exception('Number of eigenvalues must be integer, >0, < dim')

    # Find the eigenvalues of the data covariance matrix
    if evals_only:
        # This isn't called in python version.
        # Use eig function as always more efficient than eigs here
        temp_evals = np.eig(x)
    else:
        # Use eig function unless fraction of eigenvalues required is tiny
        if (N/x.shape[1]) > 0.04:
            temp_evals, temp_evec = la.eig(x)
        else:
            # Want to use eigs here, but it doesn't exist for python yet.
            # options.disp = 0
            #temp_evec, temp_evals = eigs(x, N, 'LM', options)
            temp_evals, temp_evec = la.eig(x)


    # Sort eigenvalues into descending order
    perm = np.argsort(-temp_evals)
    evals = temp_evals[perm[0:N]]
    
    if not evals_only:
        # should always come through here.
        evec = temp_evec[:, perm[0:N]]
    return evals, evec

def gradchek(w, func, grad, *args):
    """GRADCHEK Checks a user-defined gradient function using finite differences.
    
    Description
    This function is intended as a utility for other netlab functions
    (particularly optimisation functions) to use.  It enables the user to
    check whether a gradient calculation has been correctly implmented
    for a given function. GRADCHEK(W, FUNC, GRAD) checks how accurate the
    gradient  GRAD of a function FUNC is at a parameter vector X.   A
    central difference formula with step size 1.0e-6 is used, and the
    results for both gradient function and finite difference
    approximation are printed. The optional return value GRADIENT is the
    gradient calculated using the function GRAD and the return value
    DELTA is the difference between the functional and finite difference
    methods of calculating the graident.
    
    GRADCHEK(X, FUNC, GRAD, P1, P2, ...) allows additional arguments to
    be passed to FUNC and GRAD.
    
    See also
    CONJGRAD, GRADDESC, HMC, OLGD, QUASINEW, SCG
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""
    # Reasonable value for step size
    epsilon = 1.0e-6

    #func = fcnchk(func, len(args))
    #grad = fcnchk(grad, len(args))

    # Treat
    nparams = len(w)
    deltaf = np.zeros(nparams)
    step = np.zeros(nparams)
    for i in range(nparams):
        # Move a small way in the ith coordinate of w
        step[i] = 1.0
        fplus  = linef(epsilon, func, w, step, *args)
        fminus = linef(-epsilon, func, w, step, *args)
        # Use central difference formula for approximation
        deltaf[i] = 0.5*(fplus - fminus)/epsilon
        step[i] = 0.0
        
    gradient = grad(w, *args)
    print 'Checking gradient ...'
    print
    delta = gradient - deltaf
    print '   analytic   diffs     delta'
    print
    print np.c_[gradient.T, deltaf.T, delta.T]



def kmeans(centres, data, options):
    """KMEANS	Trains a k means cluster model.

    Description
    CENTRES = KMEANS(CENTRES, DATA, OPTIONS) uses the batch K-means
    algorithm to set the centres of a cluster model. The matrix DATA
    represents the data which is being clustered, with each row
    corresponding to a vector. The sum of squares error function is used.
    The point at which a local minimum is achieved is returned as
    CENTRES.  The error value at that point is returned in OPTIONS[7].
    
    [CENTRES, OPTIONS, POST, ERRLOG] = KMEANS(CENTRES, DATA, OPTIONS)
    also returns the cluster number (in a one-of-N encoding) for each
    data point in POST and a log of the error values after each cycle in
    ERRLOG.    The optional parameters have the following
    interpretations.
    
    OPTIONS[0] is set to 1 to display error values; also logs error
    values in the return argument ERRLOG. If OPTIONS[0] is set to 0, then
    only warning messages are displayed.  If OPTIONS[0] is -1, then
    nothing is displayed.
    
    OPTIONS[1] is a measure of the absolute precision required for the
    value of CENTRES at the solution.  If the absolute difference between
    the values of CENTRES between two successive steps is less than
    OPTIONS[1], then this condition is satisfied.
    
    OPTIONS[2] is a measure of the precision required of the error
    function at the solution.  If the absolute difference between the
    error functions between two successive steps is less than OPTIONS[2],
    then this condition is satisfied. Both this and the previous
    condition must be satisfied for termination.
    
    OPTIONS[13] is the maximum number of iterations; default 100.
    
    See also
    GMMINIT, GMMEM
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""
    
    #centres = np.asmatrix(centres)
    #data = np.asmatrix(data)
    ndata, data_dim = data.shape
    ncentres, dim = centres.shape

    if dim != data_dim:
        raise Exception('Data dimension does not match dimension of centres')

    if ncentres > ndata:
        raise Exception('More centres than data')

    # Sort out the options
    if options[13]:
        niters = int(options[13])
    else:
        niters = 100

    store = True
    errlog = np.zeros(niters)
    # In netlab version you can have variable number of out
    # argmuments, and choose not to look at the error log
    #store = False
    #if nargout > 3:
    #    store = 1
    #    errlog = zeros(1, niters)

    # Check if centres and posteriors need to be initialised from data
    if options[4] == 1:
        # Do the initialisation
        perm = np.random.permutation(ndata)[0:ncentres]

        # Assign first ncentres (permuted) data points as centres
        centres = data[perm, :]

    # Matrix to make unit vectors easy to construct
    idy = np.eye(ncentres)

    # Main loop of algorithm
    for n in range(niters):

        # Save old centres to check for termination
        old_centres = centres
        
        # Calculate posteriors based on existing centres
        d2 = dist2(data, centres)
        # Assign each point to nearest centre
        index = np.argmin(d2, axis=1).flatten()
        minVals = np.min(d2, axis=1)
        post = idy[index, :]
        
        num_points = np.sum(post, axis=0)
        # Adjust the centres based on new posteriors
        for j in range(ncentres):
            if num_points[j] > 0:
                centres[j,:] = data[np.nonzero(post[:, j]), :].sum(1).reshape(-1, 1).T/num_points[j]

        # Error value is total squared distance from cluster centres
        e = minVals.sum()
        if store:
            errlog[n] = e
        if options[0] > 0:
            print 'Cycle ', n, '  Error ', e

        if n > 1:
            # Test for termination
            if np.abs(centres - old_centres).max() < options[1] and \
                    abs(old_e - e) < options[2]:
                
                options[7] = e
                return centres, post, errlog 
        old_e = e

    # If we get here, then we haven't terminated in the given number of 
    # iterations.
    options[7] = e
    if options[0] >= 0:
        print "Maximum number of iterations has been exceeded"
    # Netlab also returns options, but pass by reference
    # here means it is unecessary.
    return centres, post, errlog 

    

def linef(lambd, fn, x, d, *args):
    """LINEF	Calculate function value along a line.
    
    Description
    LINEF(LAMBDA, FN, X, D) calculates the value of the function FN at
    the point X+LAMBDA*D.  Here X is a row vector and LAMBDA is a scalar.
    
    LINEF(LAMBDA, FN, X, D, P1, P2, ...) allows additional arguments to
    be passed to FN().   This function is used for convenience in some of
    the optimisation routines.
    
    See also
    GRADCHEK, LINEMIN
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""
    
    # Check function string
    #fn = fcnchk(fn, len(args))
    return fn(x+lambd*d, *args)
    
def realmin():
    """Equivalent of realmin in MATLAB."""
    return np.finfo(float).tiny
def eps(x=None):
    """Equivalent of eps in MATLAB. Assumes the input is a float."""
    epsBase = np.finfo(float).eps
    if x == None:
        return epsBase
    else:
        E = np.floor(math.log(abs(x))/math.log(2))
        return 2**(E+np.round(math.log(epsBase)/math.log(2)))

def mrank(X, tol=None):
    """Return the rank of the matrix, equivalent to rank in MATLAB."""
    s = la.svd(X, compute_uv=False)
    if tol == None:
        tol = np.diag(X).max()*eps(float(s.max()))
    return (s>tol).sum()

def realmax():
    """Equivalent of realmax in MATLAB."""
    return np.finfo(float).max

def flops(val=None):
    """Dummy function for allowing flops to be called in NETLAB code."""
    return 0

def clc():
    """CLC clears the console for use in NETLAB demos to replace the
    MATLAB clc command."""
    import os
    if os.name == "posix":
        os.system('clear')
    elif os.name in ("nt", "dos", "ce"):
        os.system('cls')
    else:
        print "\n"*100

def dem2ddat(ndata):
    """DEM2DDAT Generates two dimensional data for demos.

	Description
	The data is drawn from three spherical Gaussian distributions with
	priors 0.3, 0.5 and 0.2; centres (2, 3.5), (0, 0) and (0,2); and
	standard deviations 0.2, 0.5 and 1.0.  DATA = DEM2DDAT(NDATA)
	generates NDATA points.

	[DATA, C] = DEM2DDAT(NDATA) also returns a matrix containing the
	centres of the Gaussian distributions.

	See also
	DEMGMM1, DEMKMEAN, DEMKNN1

	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

    input_dim = 2

    # Fix seed for reproducible results
    np.random.seed(42)

    # Generate mixture of three Gaussians in two dimensional space
    data = np.random.randn(ndata, input_dim)
    #data = np.asmatrix(data)

    # Priors for the three clusters
    prior = np.array([0.3, 0.5, 0.2])

    # Cluster centres
    c = np.array([[2.0, 3.5], [0.0, 0.0], [0.0, 2.0]])
    # c = np.asmatrix(c)
    # Cluster standard deviations
    sd  = np.array([0.2, 0.5, 1.0])
    #sd = np.asmatrix(sd)
    # Put first cluster at (2, 3.5)
    data[0:int(prior[0]*ndata), 0] = data[0:int(prior[0]*ndata), 0] * 0.2 + c[0,0]
    data[0:int(prior[0]*ndata), 1] = data[0:prior[0]*ndata, 1] * 0.2 + c[0,1]

    # Leave second cluster at (0,0)
    data[int(prior[0]*ndata + 1):int((prior[1]+prior[0])*ndata), :] = \
        data[int(prior[0]*ndata + 1):int((prior[1]+prior[0])*ndata), :] * 0.5

    # Put third cluster at (0,2)
    data[int((prior[0]+prior[1])*ndata +1):ndata, 1] = \
	data[int((prior[0]+prior[1])*ndata+1):ndata, 1] + c[2, 1]
    return data, c, prior, sd


def rosen(x):
    """ROSEN	Calculate Rosenbrock's function.

    Description
    Y = ROSEN(X) computes the value of Rosenbrock's function at each row
    of X, which should have two columns.
    
    See also
    DEMOPT1, ROSEGRAD
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""

    # Calculate value of Rosenbrock's function: x should be nrows by 2 columns
    
    if x.ndim==1:
        return 100 * pow(x[1] - x[0]*x[0], 2) + pow((1.0 - x[0]), 2)
    elif x.ndim==2:
        return 100 * pow(x[:,1] - x[:,0]*x[:,0], 2) + pow((1.0 - x[:,0]), 2)
    else:
        raise Exception("Dimension of input x must be 1 or 2.")
    
def rosegrad(x):
    """ROSEGRAD Calculate gradient of Rosenbrock's function.
    
    Description
    G = ROSEGRAD(X) computes the gradient of Rosenbrock's function at
    each row of X, which should have two columns.
    
    See also
    DEMOPT1, ROSEN
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""
    
    # Return gradient of Rosenbrock's test function

    if x.ndim==1:
        g = np.zeros(2)
        g[0] = -400 * np.dot((x[1] - x[0]*x[0]), x[0]) - 2 * (1 - x[0])
        g[1] = 200 * (x[1] - x[0]*x[0])

    elif x.ndim==2:
        nrows = x.shape[0]
        g = np.zeros((nrows,2))
        g[:,0] = -400 * np.dot((x[:,1] - x[:,0]*x[:, 0]), x[:,0]) - 2 * (1 - x[:,0])
        g[:,1] = 200 * (x[:,1] - x[:,0]*x[:,0])
    else:
        raise Exception("Dimension of input x must be 1 or 2.")

    return g

def foptions():
    """FOPTIONS Sets default parameters for optimisation routines
       For compatibility with MATLAB's foptions()
       
       Copyright (c) Dharmesh Maniyar, Ian T. Nabney (2004)
       and Neil D. Lawrence (2009) (translation to python)"""
       
  
    opt_vect = np.zeros(18)
    opt_vect[1] = 1e-4
    opt_vect[2] = 1e-4
    opt_vect[3] = 1e-6
    opt_vect[15] = 1e-8
    opt_vect[16] = 0.1
    return opt_vect

def hintmat(w):
    """HINTMAT Evaluates the coordinates of the patches for a Hinton diagram.
    
    Description
    [xvals, yvals, color] = hintmat(w)
    takes a matrix W and returns coordinates XVALS, YVALS for the
    patches comrising the Hinton diagram, together with a vector COLOR
    labelling the color (black or white) of the corresponding elements
    according to their sign.
    
    See also
    HINTON
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""

    # Set scale to be up to 0.9 of maximum absolute weight value, where scale
    # defined so that area of box proportional to weight value.

    w = flipud(w)
    nrows, ncols = w.shape

    scale = 0.45*np.sqrt(np.abs(w)/abs(w).max())
    scale = scale.flatten()
    color = int(w.flatten()>0)

    delx = 1
    dely = 1
    X, Y = mgrid[0.5*delx:(ncols-0.5*delx):delx, 0.5*dely:(nrows-0.5*dely):dely]

    # Now convert from matrix format to column vector format, and then duplicate
    # columns with appropriate offsets determined by normalized weight magnitudes. 

    xtemp = X.flatten()
    ytemp = Y.flatten()

    xvals = np.r_[xtemp-delx*scale, xtemp+delx*scale, xtemp+delx*scale, xtemp-delx*scale]
    yvals = np.r_[ytemp-dely*scale, ytemp-dely*scale, ytemp+dely*scale, ytemp+dely*scale]
    return xvals, yvals, color

# def hinton(w):
#     """HINTON	Plot Hinton diagram for a weight matrix.

# 	Description

# 	HINTON(W) takes a matrix W and plots the Hinton diagram.

# 	H = HINTON(NET) also returns the figure handle H which can be used,
# 	for instance, to delete the  figure when it is no longer needed.

# 	To print the figure correctly in black and white, you should call
# 	SET(H, 'INVERTHARDCOPY', 'OFF') before printing.

# 	See also
# 	DEMHINT, HINTMAT, MLPHINT


# 	Copyright (c) Ian T Nabney (1996-2001)
#         and Neil D. Lawrence (2009) (translation to python)"""

#     # Set scale to be up to 0.9 of maximum absolute weight value, where scale
#     # defined so that area of box proportional to weight value.

#     # Use no more than 640x480 pixels
#     xmax = 640; ymax = 480

#     # Offset bottom left hand corner
#     x01 = 40; y01 = 40
#     x02 = 80; y02 = 80

#     # Need to allow 5 pixels border for window frame: but 30 at top
#     border = 5
#     top_border = 30

#     ymax = ymax - top_border
#     xmax = xmax - border

#     # First layer
    
#     xvals, yvals, color = hintmat(w)
#     # Try to preserve aspect ratio approximately
#     if 8*w.shape[0] < 6*w.shape[1]:
#         delx = xmax; dely = xmax*w.shape[0]/w.shape[1]
#     else:
#         delx = ymax*w.shape[1]/w.shape[0]; dely = ymax;

#     h = pp.figure('Color', [0.5, 0.5, 0.5], name='Hinton diagram', numbertitle='off', colormap=[[0, 0, 0],[1, 1, 1], units='pixels', position=[x01, y01, delx, dely])
#     pp.setp(gca(), visible='off', position=[0, 0, 1, 1])
#     pp.hold(True)
#     pp.patch(xvals.T, yvals.T, color.T, edgecolor='none')
#     pp.axis('equal')



def histp(x, xmin, xmax, nbins):
    """HISTP	Histogram estimate of 1-dimensional probability distribution.
    
    Description
    
    HISTP(X, XMIN, XMAX, NBINS) takes a column vector X  of data values
    and generates a normalized histogram plot of the  distribution. The
    histogram has NBINS bins lying in the range XMIN to XMAX.
    
    H = HISTP(...) returns a vector of patch handles.
    
    See also
    DEMGAUSS
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""


    pdb, bins, patches = pp.hist(x, nbins, range=(xmin, xmax), normed=True)
    return patches



def neterr(w, net, x, t, returnDataPrior=False):
    """NETERR	Evaluate network error function for generic optimizers
    
    Description
    
    E = NETERR(W, NET, X, T) takes a weight vector W and a network data
    structure NET, together with the matrix X of input vectors and the
    matrix T of target vectors, and returns the value of the error
    function evaluated at W.
    
    [E, VARARGOUT] = NETERR(W, NET, X, T) also returns any additional
    return values from the error function.
    
    See also
    NETGRAD, NETHESS, NETOPT
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""

    net.unpak(w)
    return net.err(x, t, returnDataPrior)


def netgrad(w, net, x, t, returnDataPrior = False):
    """NETGRAD Evaluate network error gradient for generic optimizers
    
    Description
    
    G = NETGRAD(W, NET, X, T) takes a weight vector W and a network data
    structure NET, together with the matrix X of input vectors and the
    matrix T of target vectors, and returns the gradient of the error
    function evaluated at W.
    
    See also
    MLP, NETERR, NETOPT
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""

    net.unpak(w)
    return net.grad(x, t, returnDataPrior)

def netopt(net, options, x, t, alg, returnOptions=False):
    """NETOPT	Optimize the weights in a network model. 
    
    Description
    
    NETOPT is a helper function which facilitates the training of
    networks using the general purpose optimizers as well as sampling
    from the posterior distribution of parameters using general purpose
    Markov chain Monte Carlo sampling algorithms. It can be used with any
    function that searches in parameter space using error and gradient
    functions.
    
    [NET, OPTIONS] = NETOPT(NET, OPTIONS, X, T, ALG) takes a network
    data structure NET, together with a vector OPTIONS of parameters
    governing the behaviour of the optimization algorithm, a matrix X of
    input vectors and a matrix T of target vectors, and returns the
    trained network as well as an updated OPTIONS vector. The string ALG
    determines which optimization algorithm (CONJGRAD, QUASINEW, SCG,
    etc.) or Monte Carlo algorithm (such as HMC) will be used.
    
    [NET, OPTIONS, VARARGOUT] = NETOPT(NET, OPTIONS, X, T, ALG) also
    returns any additional return values from the optimisation algorithm.
    
    See also
    NETGRAD, BFGS, CONJGRAD, GRADDESC, HMC, SCG
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""


    # Extract weights from network as single vector
    w = net.pak()

    args = (neterr, w, options, netgrad, (net, x, t))
    # Carry out optimisation
    w = alg(*args)[0]

    if returnOptions:
        options = s[1]

        # If there are additional arguments, extract them
        nextra = nargout - 2
        if nextra > 0:
            for i in range(nextra):
                varargout[i] = s[i+2]
    # Pack the weights back into the network
    net.unpak(w)
    return net


def maxitmess():
    """MAXITMESS Create a standard error message when training reaches max. iterations.

    Description
    S = MAXITMESS returns a standard string that it used by training
    algorithms when the maximum number of iterations (as specified in
    OPTIONS(14) is reached.
    
    See also
    CONJGRAD, GLMTRAIN, GMMEM, GRADDESC, GTMEM, KMEANS, OLGD, QUASINEW, SCG


    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""

    return 'Maximum number of iterations has been exceeded'


def olgd(net, options, x, t, returnFlog=False, returnPoint=False):
    """OLGD	On-line gradient descent optimization.

	Description
	[NET, OPTIONS, ERRLOG, POINTLOG] = OLGD(NET, OPTIONS, X, T) uses  on-
	line gradient descent to find a local minimum of the error function
	for the network NET computed on the input data X and target values T.
	A log of the error values after each cycle is (optionally) returned
	in ERRLOG, and a log of the points visited is (optionally) returned
	in POINTLOG. Because the gradient is computed on-line (i.e. after
	each pattern) this can be quite inefficient in Matlab.

	The error function value at final weight vector is returned in
	OPTIONS[7].

	The optional parameters have the following interpretations.

	OPTIONS[0] is set to 1 to display error values; also logs error
	values in the return argument ERRLOG, and the points visited in the
	return argument POINTSLOG.  If OPTIONS[0] is set to 0, then only
	warning messages are displayed.  If OPTIONS[0] is -1, then nothing is
	displayed.

	OPTIONS[1] is the precision required for the value of X at the
	solution. If the absolute difference between the values of X between
	two successive steps is less than OPTIONS[1], then this condition is
	satisfied.

	OPTIONS[2] is the precision required of the objective function at the
	solution.  If the absolute difference between the error functions
	between two successive steps is less than OPTIONS[2], then this
	condition is satisfied. Both this and the previous condition must be
	satisfied for termination. Note that testing the function value at
	each iteration roughly halves the speed of the algorithm.

	OPTIONS[4] determines whether the patterns are sampled randomly with
	replacement. If it is 0 (the default), then patterns are sampled in
	order.

	OPTIONS[5] determines if the learning rate decays.  If it is 1 then
	the learning rate decays at a rate of 1/T.  If it is 0 (the default)
	then the learning rate is constant.

	OPTIONS[8] should be set to 1 to check the user defined gradient
	function.

	OPTIONS[9] returns the total number of function evaluations
	(including those in any line searches).

	OPTIONS[10] returns the total number of gradient evaluations.

	OPTIONS[13] is the maximum number of iterations (passes through the
	complete pattern set); default 100.

	OPTIONS[16] is the momentum; default 0.5.

	OPTIONS[17] is the learning rate; default 0.01.

	See also
	GRADDESC


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

    #  Set up the options.
    if len(options) < 18:
        raise Error('Options vector too short')

    if options[13]:
        niters = options[13]
    else:
        niters = 100

    # Learning rate: must be positive
    if options[17] > 0:
        eta = options[17]
    else:
        eta = 0.01
    # Save initial learning rate for annealing
    lr = eta
# Momentum term: allow zero momentum
    if options[16] >= 0:
        mu = options[16]
    else:
        mu = 0.5


    # Extract initial weights from the network
    w = net.pak()

    display = options[0]

    # Work out if we need to compute f at each iteration.
    # Needed if display results or if termination
    # criterion requires it.
    fcneval = display or options[2]

#  Check gradients
    if options[8]:
        gradchek(w, net.err, net.grad, (x, t))

    dwold = np.zeros(len(w))
    fold = 0 # Must be initialised so that termination test can be performed
    ndata = x.shape[0]

    if fcneval:
        fnew = neterr(w, net, x, t)
        options[9] = options[9] + 1
        fold = fnew

    if returnFlog:
        errlog = [fnew]
    else:
        errlog = []
    if returnPoint:
        pointlog = [w]
    else:
        pointlog = []
    j = 0
    #  Main optimization loop.
    while j < niters:
        wold = w
        if options[4]:
            # Randomise order of pattern presentation: with replacement
            pnum = np.random.randint(0, ndata, ndata);
        else:
            pnum = np.arange(ndata)
        for k in range(ndata):
            ind = pnum[k]
            grad = netgrad(w, net, x[ind:ind+1, :], t[ind:ind+1, :])
            if options[5]:
                # Let learning rate decrease as 1/t
                lr = eta/float(j*ndata + k + 1)
            dw = mu*dwold - lr*grad
            w =  w + dw
            dwold = dw
        options[10] = options[10] + 1  # Increment gradient evaluation count
        if fcneval:
            fold = fnew
            fnew = neterr(w, net, x, t)
            options[9] = options[9] + 1
        j = j + 1;
        if display:
            print 'Iteration  ', j, '  Error ', fnew
        if returnFlog:
            # Store relevant variables
            if j >= len(errlog):
                errlog.append(fnew)		# Current function value
            else:
                errlog[j] = fnew
        if returnPoint:
            if j >= len(pointlog):
                pointlog.append(x)           # Current position
            else:
                pointlog[j] = x              
        if (np.max(np.abs(w - wold)) < options[1] and abs(fnew - fold) < options[2]):
            # Termination criteria are met
            options[7] = fnew
            net.unpak(w)
            return errlog, pointlog

    if fcneval:
        options[7] = fnew
    else:
        # Return error on entire dataset
        options[7] = neterr(w, net, x, t)
        options[9] = options[9] + 1
    if (options[0] >= 0):
        print maxitmess()

    net.unpak(w)

    return errlog, pointlog

def minbrack(f, a, b, fa, *optargs):
    """MINBRACK Bracket a minimum of a function of one variable.

    Description
    BRMIN, BRMID, BRMAX, NUMEVALS] = MINBRACK(F, A, B, FA) finds a
    bracket of three points around a local minimum of F.  The function F
    must have a one dimensional domain. A < B is an initial guess at the
    minimum and maximum points of a bracket, but MINBRACK will search
    outside this interval if necessary. The bracket consists of three
    points (in increasing order) such that F(BRMID) < F(BRMIN) and
    F(BRMID) < F(BRMAX). FA is the value of the function at A: it is
    included to avoid unnecessary function evaluations in the
    optimization routines. The return value NUMEVALS is the number of
    function evaluations in MINBRACK.
    
    MINBRACK(F, A, B, FA, P1, P2, ...) allows additional arguments to be
    passed to F
    
    See also
    LINEMIN, LINEF
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""

    # Check function string
    #f = fcnchk(f, length(varargin));

    # Value of golden section (1 + sqrt(5))/2.0
    phi = 1.6180339887499

    # Initialise count of number of function evaluations
    num_evals = 0

    # A small non-zero number to avoid dividing by zero in quadratic interpolation
    TINY = 1.e-10

    # Maximal proportional step to take: don't want to make this too big
    # as then spend a lot of time finding the minimum inside the bracket
    max_step = 10.0

    fb = f(b, *optargs)
    num_evals = num_evals + 1

    # Assume that we know going from a to b is downhill initially 
    # (usually because gradf(a) < 0).
    if (fb > fa):
        # Minimum must lie between a and b: do golden section until we find point
        # low enough to be middle of bracket
        c = b
        b = a + (c-a)/phi
        fb = f(b, *optargs)
        num_evals = num_evals + 1
        while (fb > fa):
            c = b
            b = a + (c-a)/phi
            fb = f(b, *optargs)
            num_evals = num_evals + 1
    else:  
        # There is a valid bracket upper bound greater than b
        c = b + phi*(b-a)
        fc = f(c, *optargs)
        num_evals = num_evals + 1
        bracket_found = 0

        while (fb > fc):
            # Do a quadratic interpolation (i.e. to minimum of quadratic)
            r = (b-a)*(fb-fc)
            q = (b-c)*(fb-fa)
            u = b - ((b-c)*q - (b-a)*r)/(2.0*(np.sign(q-r)*max([abs(q-r), TINY])));
            ulimit = b + max_step*(c-b)

            if (np.multiply((b-u).T,(u-c)) > 0.0):
                # Interpolant lies between b and c
                fu = f(u, *optargs)
                num_evals = num_evals + 1
                if (fu < fc):
                    # Have a minimum between b and c
                    br_min = b
                    br_mid = u
                    br_max = c
                    return br_min, br_mid, br_max, num_evals

                elif (fu > fb):
                    # Have a minimum between a and u
                    br_min = a
                    br_mid = c
                    br_max = u
                    return br_min, br_mid, br_max, num_evals

                # Quadratic interpolation didn't give a bracket, so take a golden step
                u = c + phi*(c-b)
            elif np.dot((c-u).T,(u-ulimit)) > 0.0:
                # Interpolant lies between c and limit
                fu = f(u, *optargs)
                num_evals = num_evals + 1
                if (fu < fc):
                    # Move bracket along, and then take a golden section step
                    b = c
                    c = u
                    u = c + phi*(c-b)
                else:
                    bracket_found = 1
            elif (np.dot((u-ulimit).T,(ulimit-c)) >= 0.0):
                # Limit parabolic u to maximum value
                u = ulimit
            else:
                # Reject parabolic u and use golden section step
                u = c + phi*(c-b)
            if not bracket_found:
                fu = f(u, *optargs)
                num_evals = num_evals + 1
            a = b
            b = c 
            c = u
            fa = fb 
            fb = fc
            fc = fu
    
    br_mid = b
    if (a < c):
        br_min = a
        br_max = c
    else:
        br_min = c 
        br_max = a

    return br_min, br_mid, br_max, num_evals

def linemin(f, pt, dir, fpt, options, *optargs):
    """LINEMIN One dimensional minimization.
    
    Description
    [X, OPTIONS] = LINEMIN(F, PT, DIR, FPT, OPTIONS) uses Brent's
    algorithm to find the minimum of the function F(X) along the line DIR
    through the point PT.  The function value at the starting point is
    FPT.  The point at which F has a local minimum is returned as X.  The
    function value at that point is returned in OPTIONS(8).
    
    LINEMIN(F, PT, DIR, FPT, OPTIONS, P1, P2, ...) allows  additional
    arguments to be passed to F().
    
    The optional parameters have the following interpretations.
    
    OPTIONS[0] is set to 1 to display error values.
    
    OPTIONS[1] is a measure of the absolute precision required for the
    value of X at the solution.
    
    OPTIONS[2] is a measure of the precision required of the objective
    function at the solution.  Both this and the previous condition must
    be satisfied for termination.
    
    OPTIONS[13] is the maximum number of iterations; default 100.
    
    See also
    CONJGRAD, MINBRACK, QUASINEW
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""

    # Set up the options.
    if options[13]:
        niters = options[13]
    else:
        niters = 100;

    options[9] = 0 # Initialise count of function evaluations

    display = options[0]

    # Check function string
    #f = fcnchk(f, length(varargin));
    
    # Value of golden section (1 + sqrt(5))/2.0
    phi = 1.6180339887499
    cphi = 1 - 1/phi
    TOL = math.sqrt(eps())	# Maximal fractional precision
    TINY = 1.0e-10         # Can't use fractional precision when minimum is at 0

    # Bracket the minimum
    br_min, br_mid, br_max, num_evals = minbrack(linef, 0.0, 1.0, fpt, f, pt, dir, *optargs)
    options[9] = options[9] + num_evals  # Increment number of fn. evals
                                            # No gradient evals in minbrack

    # Use Brent's algorithm to find minimum
    # Initialise the points and function values
    w = br_mid   	# Where second from minimum is
    v = br_mid   	# Previous value of w
    x = v   	# Where current minimum is
    e = 0.0 	# Distance moved on step before last
    fx = linef(x, f, pt, dir, *optargs)
    options[9] = options[9] + 1
    fv = fx
    fw = fx
    
    for n in range(niters):
        xm = 0.5*(br_min+br_max)  # Middle of bracket
        # Make sure that tolerance is big enough
        tol1 = TOL * (np.max(np.abs(x))) + TINY
        # Decide termination on absolute precision required by options(2)
        if (np.max(np.abs(x - xm)) <= options[1] and br_max-br_min < 4*options[1]):
            options[7] = fx
            return x
        # Check if step before last was big enough to try a parabolic step.
        # Note that this will fail on first iteration, which must be a golden
        # section step.
        if (np.max(np.abs(e)) > tol1):
            # Construct a trial parabolic fit through x, v and w
            r = (fx - fv) * (x - w)
            q = (fx - fw) * (x - v)
            p = (x - v)*q - (x - w)*r
            q = 2.0 * (q - r)
            if (q > 0.0):
                p = -p
            q = np.abs(q)
            # Test if the parabolic fit is OK
            if (abs(p) >= abs(0.5*q*e) or p <= q*(br_min-x) or p >= q*(br_max-x)):
                # No it isn't, so take a golden section step
                if (x >= xm):
                    e = br_min-x
                else:
                    e = br_max-x
                d = cphi*e;
            else:
                # Yes it is, so take the parabolic step
                e = d
                d = p/q
                u = x+d
                if (u-br_min < 2*tol1 or br_max-u < 2*tol1):
                    d = np.sign(xm-x)*tol1
        else:
            # Step before last not big enough, so take a golden section step
            if (x >= xm):
                e = br_min - x
            else:
                e = br_max - x
            d = cphi*e;
      # Make sure that step is big enough
        if (abs(d) >= tol1):
            u = x+d
        else:
            u = x + np.sign(d)*tol1
        # Evaluate function at u
        fu = linef(u, f, pt, dir, *optargs)
        options[9] = options[9] + 1
      # Reorganise bracket
        if (fu <= fx):
            if (u >= x):
                br_min = x
            else:
                br_max = x
            v = w 
            w = x 
            x = u
            fv = fw
            fw = fx 
            fx = fu
        else:
            if (u < x):
                br_min = u
            else:
                br_max = u
            if (fu <= fw or w == x):
                v = w
                w = u
                fv = fw
                fw = fu
            elif (fu <= fv or v == x or v == w):
                v = u
                fv = fu
        if (display == 1):
            print 'Cycle ', n, ' Error ', fx
    options[7] = fx
    return x

def conjgrad(f, x, options, gradf, optargs=[], 
             returnFlog=False, 
             returnPoint=False):
    """CONJGRAD Conjugate gradients optimization.

	Description
	[X, OPTIONS, FLOG, POINTLOG] = CONJGRAD(F, X, OPTIONS, GRADF) uses a
	conjugate gradients algorithm to find the minimum of the function
	F(X) whose gradient is given by GRADF(X).  Here X is a row vector and
	F returns a scalar value.  The point at which F has a local minimum
	is returned as X.  The function value at that point is returned in
	OPTIONS[7].  A log of the function values after each cycle is
	(optionally) returned in FLOG, and a log of the points visited is
	(optionally) returned in POINTLOG.

	CONJGRAD(F, X, OPTIONS, GRADF, P1, P2, ...) allows  additional
	arguments to be passed to F() and GRADF().

	The optional parameters have the following interpretations.

	OPTIONS[0] is set to 1 to display error values; also logs error
	values in the return argument ERRLOG, and the points visited in the
	return argument POINTSLOG.  If OPTIONS(1) is set to 0, then only
	warning messages are displayed.  If OPTIONS(1) is -1, then nothing is
	displayed.

	OPTIONS[1] is a measure of the absolute precision required for the
	value of X at the solution.  If the absolute difference between the
	values of X between two successive steps is less than OPTIONS(2),
	then this condition is satisfied.

	OPTIONS[2] is a measure of the precision required of the objective
	function at the solution.  If the absolute difference between the
	objective function values between two successive steps is less than
	OPTIONS[2], then this condition is satisfied. Both this and the
	previous condition must be satisfied for termination.

	OPTIONS[8] is set to 1 to check the user defined gradient function.

	OPTIONS[9] returns the total number of function evaluations
	(including those in any line searches).

	OPTIONS[10] returns the total number of gradient evaluations.

	OPTIONS[13] is the maximum number of iterations; default 100.

	OPTIONS[14] is the precision in parameter space of the line search;
	default 1E-4.

	See also
	GRADDESC, LINEMIN, MINBRACK, QUASINEW, SCG


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""
    
    #  Set up the options.
    if len(options) < 18:
        raise Exception('Options vector too short')

    if options[13]:
        niters = options[13]
    else:
        niters = 100

    # Set up options for line search
    line_options = foptions()

    # Need a precise line search for success
    if options[14] > 0:
        line_options[1] = options[14]
    else:
        line_options[1] = 1e-4

    display = options[0]

    # Next two lines allow conjgrad to work with expression strings
    # f = fcnchk(f, length(varargin));
    # gradf = fcnchk(gradf, length(varargin));

    #  Check gradients
    if options[8]:
        gradchek(x, f, gradf, *optargs);

    options[9] = 0
    options[10] = 0
    nparams = len(x)
    fnew = f(x, *optargs)
    options[9] = options[9] + 1
    gradnew = gradf(x, *optargs)
    options[10] = options[10] + 1
    d = -gradnew		# Initial search direction
    br_min = 0
    br_max = 1.0	# Initial value for maximum distance to search along
    tol = math.sqrt(eps())

    j = 0
    if returnFlog:
        flog = [fnew]
    else:
        flog = []
    if returnPoint:
        pointlog = [x]
    else:
        pointlog = []

    while (j < niters):

        xold = x
        fold = fnew
        gradold = gradnew

        gg = np.dot(gradold, gradold)
        if (gg == 0.0):
            # If the gradient is zero then we are done.
            options[7] = fnew
            return x, flog, pointlog

        # This shouldn't occur, but rest of code depends on d being downhill
        if (np.dot(gradnew,d) > 0):
            d = -d
            if options[0] >= 0:
                print 'Warning: search direction uphill in conjgrad'

        line_sd = d/la.norm(d)
        lmin = linemin(f, xold, line_sd, fold, line_options, *optargs)
        options[9] = options[9] + line_options[9]
        options[10] = options[10] + line_options[10]
        # Set x and fnew to be the actual search point we have found
        x = xold + lmin * line_sd
        fnew = line_options[7]

        # Check for termination
        if np.max(np.abs(x - xold)) < options[1] and abs(fnew - fold) < options[2]:
            options[7] = fnew
            return x, flog, pointlog

        gradnew = gradf(x, *optargs)
        options[10] = options[10] + 1

        # Use Polak-Ribiere formula to update search direction
        gamma = np.dot(gradnew - gradold, gradnew)/gg
        d = (d*gamma) - gradnew

        j = j + 1
        if (display > 0):
            print 'Cycle ', j, '  Function ', line_options[7]

        if returnFlog:
            # Store relevant variables
            if j >= len(flog):
                flog.append(fnew)		# Current function value
            else:
                flog[j] = fnew
        if returnPoint:
            if j >= len(pointlog):
                pointlog.append(x)           # Current position
            else:
                pointlog[j] = x              

    # If we get here, then we haven't terminated in the given number of 
    # iterations.
    options[7] = fold
    if (options[0] >= 0):
        print maxitmess()
    
    return x, flog, pointlog


def graddesc(f, x, options, gradf, optargs=[], 
             returnFlog=False, 
             returnPoint=False):
    """GRADDESC Gradient descent optimization.
    
    Description
    [X, OPTIONS, FLOG, POINTLOG] = GRADDESC(F, X, OPTIONS, GRADF) uses
    batch gradient descent to find a local minimum of the function  F(X)
    whose gradient is given by GRADF(X). A log of the function values
    after each cycle is (optionally) returned in ERRLOG, and a log of the
    points visited is (optionally) returned in POINTLOG.
    
    Note that X is a row vector and F returns a scalar value.  The point
    at which F has a local minimum is returned as X.  The function value
    at that point is returned in OPTIONS(8).

    GRADDESC(F, X, OPTIONS, GRADF, P1, P2, ...) allows  additional
    arguments to be passed to F() and GRADF().
    
    The optional parameters have the following interpretations.
    
    OPTIONS[0] is set to 1 to display error values; also logs error
    values in the return argument ERRLOG, and the points visited in the
    return argument POINTSLOG. If OPTIONS[0] is set to 0, then only
    warning messages are displayed.  If OPTIONS[0] is -1, then nothing is
    displayed.
    
    OPTIONS[1] is the absolute precision required for the value of X at
    the solution.  If the absolute difference between the values of X
    between two successive steps is less than OPTIONS[1], then this
    condition is satisfied.
    
    OPTIONS[2] is a measure of the precision required of the objective
    function at the solution.  If the absolute difference between the
    objective function values between two successive steps is less than
    OPTIONS[2], then this condition is satisfied. Both this and the
    previous condition must be satisfied for termination.
    
    OPTIONS[6] determines the line minimisation method used.  If it is
    set to 1 then a line minimiser is used (in the direction of the
    negative gradient).  If it is 0 (the default), then each parameter
    update is a fixed multiple (the learning rate) of the negative
    gradient added to a fixed multiple (the momentum) of the previous
    parameter update.
    
    OPTIONS[8] should be set to 1 to check the user defined gradient
    function GRADF with GRADCHEK.  This is carried out at the initial
    parameter vector X.

    OPTIONS[9] returns the total number of function evaluations
    (including those in any line searches).
    
    OPTIONS[10] returns the total number of gradient evaluations.
    
    OPTIONS[13] is the maximum number of iterations; default 100.
    
    OPTIONS[14] is the precision in parameter space of the line search;
    default FOPTIONS[1].
    
    OPTIONS[16] is the momentum; default 0.5.  It should be scaled by the
    inverse of the number of data points.
    
    OPTIONS[17] is the learning rate; default 0.01.  It should be scaled
    by the inverse of the number of data points.
    
    See also
    CONJGRAD, LINEMIN, OLGD, MINBRACK, QUASINEW, SCG
    

    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""
    
    #  Set up the options.
    if len(options) < 18:
        raise Exception('Options vector too short')

    if (options[13]):
        niters = int(options[13])
    else:
        niters = 100

    line_min_flag = False # Flag for line minimisation option
    if round(options[6]) == 1:
        # Use line minimisation
        line_min_flag = True
        # Set options for line minimiser
        line_options = foptions()
        if options[14] > 0:
            line_options[1] = options[14]
    else:
        # Learning rate: must be positive
        if (options[17] > 0):
            eta = options[17]
        else:
            eta = 0.01
        # Momentum term: allow zero momentum
        if (options[16] >= 0):
            mu = options[16]
        else:
            mu = 0.5

    # Check function string
    #f = fcnchk(f, length(varargin))
    #gradf = fcnchk(gradf, length(varargin));

    # Display information if options[0] > 0
    display = options[0] > 0

    # Work out if we need to compute f at each iteration.
    # Needed if using line search or if display results or if termination
    # criterion requires it.
    fcneval = (options[6] or display or options[2])

    #  Check gradients
    if (options[8] > 0):
        gradchek(x, f, gradf, *optargs)

    dxold = np.zeros(len(x))
    xold = x
    fold = 0 # Must be initialised so that termination test can be performed
    if fcneval:
        fnew = f(x, *optargs)
        options[9] = options[9] + 1
        fold = fnew

    flog = []
    pointlog = []


    #  Main optimization loop.
    for j in range(niters):
        xold = x
        grad = gradf(x, *optargs)
        options[10] = options[10] + 1  # Increment gradient evaluation counter
        if not line_min_flag:
            dx = mu*dxold - eta*grad
            x =  x + dx
            dxold = dx
            if fcneval:
                fold = fnew
                fnew = f(x, *optargs)
                options[9] = options[9] + 1
        else:
            sd = - grad/la.norm(grad)	# New search direction.
            fold = fnew
            # Do a line search: normalise search direction to have length 1
            lmin = linemin(f, x, sd, fold, line_options, *optargs)
            options[9] = options[9] + line_options[9]
            x = xold + lmin*sd
            fnew = line_options[7]
        if returnFlog:
            # Store relevant variables
            if j >= len(flog):
                flog.append(fnew)		# Current function value
            else:
                flog[j] = fnew
        if returnPoint:
            if j >= len(pointlog):
                pointlog.append(x)           # Current position
            else:
                pointlog[j] = x              
        if display:
            print 'Cycle', j, '  Function ', fnew
        if (max(abs(x - xold)) < options[1] and abs(fnew - fold) < options[2]):
            # Termination criteria are met
            options[7] = fnew
            return x, flog, pointlog

    if fcneval:
        options[7] = fnew;
    else:
        options[7] = f(x, *optargs)
        options[9] = options[9] + 1
    if options[0] >= 0:
        print maxitmess()
    return x, flog, pointlog
    

def quasinew(f, x, options, gradf, optargs=[], 
             returnFlog=False, 
             returnPoint=False):
    """QUASINEW Quasi-Newton optimization.

	Description
	[X, OPTIONS, FLOG, POINTLOG] = QUASINEW(F, X, OPTIONS, GRADF)  uses a
	quasi-Newton algorithm to find a local minimum of the function F(X)
	whose gradient is given by GRADF(X).  Here X is a row vector and F
	returns a scalar value.   The point at which F has a local minimum is
	returned as X.  The function value at that point is returned in
	OPTIONS(8). A log of the function values after each cycle is
	(optionally) returned in FLOG, and a log of the points visited is
	(optionally) returned in POINTLOG.

	QUASINEW(F, X, OPTIONS, GRADF, P1, P2, ...) allows  additional
	arguments to be passed to F() and GRADF().

	The optional parameters have the following interpretations.

	OPTIONS[0] is set to 1 to display error values; also logs error
	values in the return argument ERRLOG, and the points visited in the
	return argument POINTSLOG.  If OPTIONS(1) is set to 0, then only
	warning messages are displayed.  If OPTIONS(1) is -1, then nothing is
	displayed.

	OPTIONS[1] is a measure of the absolute precision required for the
	value of X at the solution.  If the absolute difference between the
	values of X between two successive steps is less than OPTIONS(2),
	then this condition is satisfied.

	OPTIONS[2] is a measure of the precision required of the objective
	function at the solution.  If the absolute difference between the
	objective function values between two successive steps is less than
	OPTIONS[2], then this condition is satisfied. Both this and the
	previous condition must be satisfied for termination.

	OPTIONS[8] should be set to 1 to check the user defined gradient
	function.

	OPTIONS[9] returns the total number of function evaluations
	(including those in any line searches).

	OPTIONS[10] returns the total number of gradient evaluations.

	OPTIONS[13] is the maximum number of iterations; default 100.

	OPTIONS[14] is the precision in parameter space of the line search;
	default 1E-2.

	See also
	CONJGRAD, GRADDESC, LINEMIN, MINBRACK, SCG


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""
    
    #  Set up the options.
    if len(options) < 18:
        raise Exception('Options vector too short')

    if(options[13]):
        niters = options[13]
    else:
        niters = 100;

    # Set up options for line search
    line_options = foptions()

    # Don't need a very precise line search
    if options[14] > 0:
        line_options[1] = options[14]
    else:
        line_options[1] = 1e-2  # Default
    # Minimal fractional change in f from Newton step: otherwise do a line search
    min_frac_change = 1e-4

    display = options[0]

    # Next two lines allow quasinew to work with expression strings
    # f = fcnchk(f, length(varargin));
    # gradf = fcnchk(gradf, length(varargin));

    # Check gradients
    if options[8]:
        gradchek(x, f, gradf, *optargs)

    nparams = len(x)
    fnew = f(x, *optargs)
    options[9] = options[9] + 1
    gradnew = gradf(x, *optargs)
    options[10] = options[10] + 1
    p = -gradnew		# Search direction
    hessinv = np.eye(nparams) # Initialise inverse Hessian to be identity matrix
    j = 0
    if returnFlog:
        flog = [fnew]
    else:
        flog = []
    if returnPoint:
        pointlog = [x]
    else:
        pointlog = []

    while (j < niters):

        xold = x
        fold = fnew
        gradold = gradnew

        x = xold + p
        fnew = f(x, *optargs)
        options[9] = options[9] + 1

        # This shouldn't occur, but rest of code depends on sd being downhill
        if np.dot(gradnew,p) >= 0:
            p = -p
            if options[0] >= 0:
                print 'Warning: search direction uphill in quasinew'

        # Does the Newton step reduce the function value sufficiently?
        if fnew >= fold + min_frac_change * np.dot(gradnew,p):
            # No it doesn't
            # Minimize along current search direction: must be less than Newton step
            lmin = linemin(f, xold, p, fold, line_options, *optargs)
            options[9] = options[9] + line_options[9]
            options[10] = options[10] + line_options[10]
            # Correct x and fnew to be the actual search point we have found
            x = xold + lmin * p
            p = x - xold
            fnew = line_options[7]

        # Check for termination
        if np.max(np.abs(x - xold)) < options[1] and abs(fnew - fold) < options[2]:
            options[7] = fnew
            return x, flog, pointlog 
        gradnew = gradf(x, *optargs)
        options[10] = options[10] + 1
        v = gradnew - gradold
        vdotp = np.dot(v,p)

        # Skip update to inverse Hessian if fac not sufficiently positive
        if vdotp*vdotp > eps()*np.dot(v, v)*np.dot(p, p):
            Gv = np.dot(hessinv,v)
            vGv = np.dot(v, Gv)
            u = p/vdotp - Gv/vGv
            # Use BFGS update rule
            hessinv = hessinv + np.outer(p,p)/vdotp - np.outer(Gv,Gv)/vGv + vGv*np.outer(u, u)

        p = -np.dot(hessinv, gradnew)

        j = j + 1
        if display > 0:
            print 'Cycle ', j, '  Function ', fnew

        if returnFlog:
            if j >= len(flog):
                flog.append(fnew)		# Current function value
            else:
                flog[j] = fnew
        if returnPoint:
            if j >= len(pointlog):
                pointlog.append(x)           # Current position
            else:
                pointlog[j] = x              

    # If we get here, then we haven't terminated in the given number of 
    # iterations.                
    options[7] = fold
    if options[0] >= 0:
        print maxitmess()

    return x, flog, pointlog 

def scg(f, x, options, gradf, optargs=[], 
        returnFlog=False, 
        returnPoint=False, 
        returnScale=False):
    """SCG	Scaled conjugate gradient optimization.

    Description
    [X, OPTIONS] = SCG(F, X, OPTIONS, GRADF) uses a scaled conjugate
    gradients algorithm to find a local minimum of the function F(X)
    whose gradient is given by GRADF(X).  Here X is a row vector and F
    returns a scalar value. The point at which F has a local minimum is
    returned as X.  The function value at that point is returned in
    OPTIONS[7].
    
    [X, OPTIONS, FLOG, POINTLOG, SCALELOG] = SCG(F, X, OPTIONS, GRADF)
    also returns (optionally) a log of the function values after each
    cycle in FLOG, a log of the points visited in POINTLOG, and a log of
    the scale values in the algorithm in SCALELOG.
    
    SCG(F, X, OPTIONS, GRADF, P1, P2, ...) allows additional arguments to
    be passed to F() and GRADF().     The optional parameters have the
    following interpretations.
    
    OPTIONS[0] is set to 1 to display error values; also logs error
    values in the return argument ERRLOG, and the points visited in the
    return argument POINTSLOG.  If OPTIONS[0] is set to 0, then only
    warning messages are displayed.  If OPTIONS[0] is -1, then nothing is
    displayed.
    
    OPTIONS[1] is a measure of the absolute precision required for the
    value of X at the solution.  If the absolute difference between the
    values of X between two successive steps is less than OPTIONS[1],
    then this condition is satisfied.
    
    OPTIONS[2] is a measure of the precision required of the objective
    function at the solution.  If the absolute difference between the
    objective function values between two successive steps is less than
    OPTIONS[2], then this condition is satisfied. Both this and the
    previous condition must be satisfied for termination.
    
    OPTIONS[8] is set to 1 to check the user defined gradient function.
    
    OPTIONS[9] returns the total number of function evaluations
    (including those in any line searches).
    
    OPTIONS[10] returns the total number of gradient evaluations.
    
    OPTIONS[13] is the maximum number of iterations; default 100.
    
    See also
    CONJGRAD, QUASINEW
    
    
    Copyright (c) Ian T Nabney (1996-2001)
    and Neil D. Lawrence (2009) (translation to python)"""

    #  Set up the options.
    if len(options) < 18:
        raise Exception('Options vector too short')

    if not options[13] == 0:
        niters = options[13]
    else:
        niters = 100

    display = options[0]
    gradcheck = options[8]

    # Set up strings for evaluating function and gradient
    #f = fcnchk(f, length(varargin))
    #gradf = fcnchk(gradf, length(varargin))

    nparams = len(x)
    #  Check gradients
    if gradcheck:
        gradchek(x, f, gradf, *optargs)

    sigma0 = 1.0e-4
    fold = f(x, *optargs)	# Initial function value.
    fnow = fold
    options[9] = options[9] + 1		# Increment function evaluation counter.
    gradnew = gradf(x, *optargs)	# Initial gradient.
    gradold = gradnew
    options[10] = options[10] + 1		# Increment gradient evaluation counter.
    d = -gradnew				# Initial search direction.
    success = True				# Force calculation of directional derivs.
    nsuccess = 0				# nsuccess counts number of successes.
    beta = 1.0				# Initial scale parameter.
    betamin = 1.0e-15 			# Lower bound on scale.
    betamax = 1.0e100			# Upper bound on scale.

    flog = []
    pointlog = []
    scalelog = []

    j = 0					# j counts number of iterations.
    if returnFlog:
        flog = [fold]
    if returnPoint:
        pointlog = [x]
    if returnScale:
        scalelog = []

    # Main optimization loop.
    while j < niters:

        # Calculate first and second directional derivatives.
        if success:
            mu = np.dot(d, gradnew)
            if mu >= 0:
                d = -gradnew
                mu = np.dot(d, gradnew)
            kappa = np.dot(d, d)
            if kappa < eps():
                options[7] = fnow
                return x, flog, pointlog, scalelog
    
            sigma = sigma0/math.sqrt(kappa)
            xplus = x + sigma*d
            gplus = gradf(xplus, *optargs)
            options[10] = options[10] + 1
            theta = np.dot(d, (gplus - gradnew))/sigma
            

        # Increase effective curvature and evaluate step size alpha.
        delta = theta + beta*kappa
        if delta <= 0:
            delta = beta*kappa
            beta = beta - theta/kappa

        alpha = - mu/delta
  
        # Calculate the comparison ratio.
        xnew = x + alpha*d
        fnew = f(xnew, *optargs)
        options[9] = options[9] + 1
        Delta = 2*(fnew - fold)/(alpha*mu)
        if Delta  >= 0:
            success = True
            nsuccess = nsuccess + 1
            x = xnew
            fnow = fnew
        else:
            success = False
            fnow = fold
  

        if returnFlog:
            # Store relevant variables 
            if j >= len(flog):
                flog.append(fnow)		# Current function value
            else:
                flog[j] = fnow
        if returnPoint:
            if j >= len(pointlog):
                pointlog.append(x)           # Current position
            else:
                pointlog[j] = x              
        if returnScale:
            if j >= len(scalelog):
                scalelog.append(beta)    # current scale parameter
            else:
                scalelog[j] = beta
                    
    
      
        j = j + 1
        if display > 0:
            print 'Cycle ', j, ' Error ', fnow, '  Scale ', beta

        if success:
            # Test for termination

            if np.max(np.abs(alpha*d)) < options[1] and np.max(np.abs(fnew-fold)) < options[3]:
                options[7] = fnew
                return x, flog, pointlog, scalelog

            else:
                # Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = gradf(x, *optargs)
                options[10] = options[10] + 1
                # If the gradient is zero then we are done.
                if np.dot(gradnew,gradnew) == 0:
                    options[7] = fnew
                    return x, flog, pointlog, scalelog

        # Adjust beta according to comparison ratio.
        if Delta < 0.25:
            beta = min(4.0*beta, betamax)
        if Delta > 0.75:
            beta = max(0.5*beta, betamin)

        # Update search direction using Polak-Ribiere formula, or re-start 
        # in direction of negative gradient after nparams steps.
        if nsuccess == nparams:
            d = -gradnew
            nsuccess = 0
        else:
            if success:
                gamma = np.dot(gradold - gradnew,gradnew)/(mu)
                d = gamma*d - gradnew

    # If we get here, then we haven't terminated in the given number of 
    # iterations.

    options[7] = fold
    if options[0] >= 0:
        print maxitmess()

    return x, flog, pointlog, scalelog

class netlabDistribution:
    pass

class gprior(netlabDistribution):
    def __init__(self, alpha=None, index=None):
        self.alpha = alpha
        self.index = index
    
    def __getattr__(self, name):
        return self.__dict__[name]

    def __setattr__(self, name):
        self.__dict__[name] = val
            

class netlabModel:
    def __getattr__(self, name):
        return self.__dict__[name]

    def __setattr__(self, name, val):
        self.__dict__[name] = val

    def consist(self, type=None, inputs=None, outputs=None):
        """CONSIST Check that arguments are consistent.

        Description

        ERRSTRING = CONSIST(NET, TYPE, INPUTS) takes a network data structure
        NET together with a string TYPE containing the correct network type,
        a matrix INPUTS of input vectors and checks that the data structure
        is consistent with the other arguments.  An empty string is returned
        if there is no error, otherwise the string contains the relevant
        error message.  If the TYPE string is empty, then any type of network
        is allowed.

        ERRSTRING = CONSIST(NET, TYPE) takes a network data structure NET
        together with a string TYPE containing the correct  network type, and
        checks that the two types match.

        ERRSTRING = CONSIST(NET, TYPE, INPUTS, OUTPUTS) also checks that the
        network has the correct number of outputs, and that the number of
        patterns in the INPUTS and OUTPUTS is the same.  The fields in NET
        that are used are
          type
          nin
          nout

        See also
        MLPFWD


        Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""


        # If type string is not empty
        if type != None:
            # Check that model has the correct type
            s = self.type
            if s != type:
                return "Model type '" + s + "' does not match expected type '"+ type + "'"

        # If inputs are present, check that they have correct dimension
        if inputs != None:
            data_nin = inputs.shape[1]
            if self.nin != data_nin:
                return 'Dimension of inputs ' + str(data_nin) + ' does not match number of model inputs ' + str(self.nin)

        # If outputs are present, check that they have correct dimension
        if outputs != None:
            data_nout = outputs.shape[1]
            if self.nout != data_nout:
                return 'Dimension of outputs ' + str(data_nout) + ' does not match number of model outputs ' + str(self.nout)

        # Also check that number of data points in inputs and outputs is the same
        if outputs != None and inputs != None:
            num_in = inputs.shape[0]
            num_out = outputs.shape[0]
            if num_in != num_out:
                return 'Number of input patterns ' + str(num_in) + ' does not match number of output patterns ' + str(num_out)
        return None

    def err(self, x, t):
        pass    

    def hbayes(self, hdata):
        """HBAYES	Evaluate Hessian of Bayesian error function for network.

	Description
	H = HBAYES(NET, HDATA) takes a network data structure NET together
	the data contribution to the Hessian for a set of inputs and targets.
	It returns the regularised Hessian using any zero mean Gaussian
	priors on the weights defined in NET.  In addition, if a MASK is
	defined in NET, then the entries in H that correspond to weights with
	a 0 in the mask are removed.

	[H, HDATA] = HBAYES(NET, HDATA) additionally returns the data
	component of the Hessian.

	See also
	GBAYES, GLMHESS, MLPHESS, RBFHESS


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        if self.mask != None:
            # Extract relevant entries in Hessian
            nindx_cols = self.index.shape[1]
            nmask_rows = (np.nonzero(self.mask)[0]).shape[0]
            index = (self.index(np.nonzero(np.tile(self.mask (1, nindx_cols))))).reshape(nmask_rows, nindx_cols, order = 'F').copy()

            nmask_rows = size(find(net.mask), 1);
            hdata = reshape(hdata(logical(net.mask*(net.mask.T))), nmask_rows, nmask_rows);
            nwts = nmask_rows
        else:
            nwts = self.nwts

        if self.beta != None:
            h = self.beta*hdata
        else:
            h = hdata

        if self.alpha != None:
            w = self.pak()
            if isinstance(self.alpha, types.FloatType):
                h = h + self.alpha*np.eye(self.nwts)
            else:
                if self.mask != None:
                    nindx_cols = self.index.shape[1]
                    nmask_rows = (np.nonzero(self.mask)[0]).shape[0]
                    index = (self.index(np.nonzero(np.tile(self.mask (1, nindx_cols))))).reshape(nmask_rows, nindx_cols, order = 'F').copy()
                else:
                    index = self.index
                h = h + np.diag(np.dot(index, self.alpha))
        return h, hdata

    def errbayes(self, edata, returnDataPrior=False):
        """ERRBAYES Evaluate Bayesian error function for network.
        
	Description
	E = ERRBAYES(NET, EDATA) takes a network data structure  NET together
	the data contribution to the error for a set of inputs and targets.
	It returns the regularised error using any zero mean Gaussian priors
	on the weights defined in NET.

	[E, EDATA, EPRIOR] = ERRBAYES(NET, X, T) additionally returns the
	data and prior components of the error.

	See also
	GLMERR, MLPERR, RBFERR


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Evaluate the data contribution to the error.
        if self.beta != None:
            e1 = self.beta*edata
        else:
            e1 = edata

        # Evaluate the prior contribution to the error.
        if self.alpha != None:
            w = self.pak()
            if isinstance(self.alpha, types.FloatType):
                eprior = 0.5*np.multiply(w, w).sum()
                e2 = eprior*self.alpha
            else:
                if self.mask != None:
                    nindx_cols = self.index.shape[1]
                    nmask_rows = (np.nonzero(self.mask)[0]).shape[0]
                    index = (self.index(np.nonzero(np.tile(self.mask (1, nindx_cols))))).reshape(nmask_rows, nindx_cols, order = 'F').copy()
                else:
                    index = self.index
                eprior = 0.5*(np.multiply(w,w))*index
                e2 = np.multiply(eprior, self.alpha).sum()
                
        else:
            eprior = 0
            e2 = 0
            

        e = e1 + e2
        if returnDataPrior:
            return e, edata, eprior
        else:
            return e
        


    def grad(self, x, t, returnDataPrior):
        pass

    def pak(self):
        pass

    def unpak(self, w):
        pass

class mlp(netlabModel):
    def __init__(self, nin, nhidden, nout, outfunc='linear', prior=None, beta=None):
        """MLP	Create a 2-layer feedforward network.

        Description
        NET = MLP(NIN, NHIDDEN, NOUT, FUNC) takes the number of inputs,
        hidden units and output units for a 2-layer feed-forward network,
        together with a string FUNC which specifies the output unit
        activation function, and returns a data structure NET. The weights
        are drawn from a zero mean, unit variance isotropic Gaussian, with
        varianced scaled by the fan-in of the hidden or output units as
        appropriate. This makes use of the Matlab function RANDN and so the
        seed for the random weight initialization can be  set using
        RANDN('STATE', S) where S is the seed value.  The hidden units use
        the TANH activation function.
        
        The fields in NET are
          type = 'mlp'
          nin = number of inputs
          nhidden = number of hidden units
          nout = number of outputs
          nwts = total number of weights and biases
          actfn = string describing the output unit activation function:
             'linear'
             'logistic
             'softmax'
          w1 = first-layer weight matrix
          b1 = first-layer bias vector
          w2 = second-layer weight matrix
          b2 = second-layer bias vector
         Here W1 has dimensions NIN times NHIDDEN, B1 has dimensions 1 times
        NHIDDEN, W2 has dimensions NHIDDEN times NOUT, and B2 has dimensions
        1 times NOUT.

        NET = MLP(NIN, NHIDDEN, NOUT, FUNC, PRIOR), in which PRIOR is a
        scalar, allows the field NET.ALPHA in the data structure NET to be
        set, corresponding to a zero-mean isotropic Gaussian prior with
        inverse variance with value PRIOR. Alternatively, PRIOR can consist
        of a data structure with fields ALPHA and INDEX, allowing individual
        Gaussian priors to be set over groups of weights in the network. Here
        ALPHA is a column vector in which each element corresponds to a
        separate group of weights, which need not be mutually exclusive.  The
        membership of the groups is defined by the matrix INDX in which the
        columns correspond to the elements of ALPHA. Each column has one
        element for each weight in the matrix, in the order defined by the
        function MLPPAK, and each element is 1 or 0 according to whether the
        weight is a member of the corresponding group or not. A utility
        function MLPPRIOR is provided to help in setting up the PRIOR data
        structure.

        NET = MLP(NIN, NHIDDEN, NOUT, FUNC, PRIOR, BETA) also sets the
        additional field NET.BETA in the data structure NET, where beta
        corresponds to the inverse noise variance.
        
        See also
        MLPPRIOR, MLPPAK, MLPUNPAK, MLPFWD, MLPERR, MLPBKP, MLPGRAD


        Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        self.mask = None
        self.type = 'mlp'
        self.nin = nin
        self.nhidden = nhidden
        self.nout = nout
        self.nwts = (nin + 1)*nhidden + (nhidden + 1)*nout

        outfns = ['linear', 'logistic', 'softmax']

        if outfunc not in outfns:
            raise Exception('Undefined output function. Exiting.')
        else:
            self.outfn = outfunc

        if prior != None:
            if isinstance(prior, netlabDistribution):
                self.alpha = prior.alpha
                self.index = prior.index
            elif isinstance(prior, types.FloatType):
                self.alpha = prior
            else:
                raise Exception('prior must be a scalar or a structure')
        else:
            self.alpha = None
            
        self.w1 = np.random.randn(nin, nhidden)/math.sqrt(nin + 1)
        self.b1 = np.random.randn(1, nhidden)/math.sqrt(nin + 1)
        self.w2 = np.random.randn(nhidden, nout)/math.sqrt(nhidden + 1)
        self.b2 = np.random.randn(1, nout)/math.sqrt(nhidden + 1)

        if not beta is None:
            self.beta = beta
        else:
            self.beta = None

    def init(self, prior):
        """MLPINIT Initialise the weights in a 2-layer feedforward network.

	Description

	NET = MLPINIT(NET, PRIOR) takes a 2-layer feedforward network NET and
	sets the weights and biases by sampling from a Gaussian distribution.
	If PRIOR is a scalar, then all of the parameters (weights and biases)
	are sampled from a single isotropic Gaussian with inverse variance
	equal to PRIOR. If PRIOR is a data structure of the kind generated by
	MLPPRIOR, then the parameters are sampled from multiple Gaussians
	according to their groupings (defined by the INDEX field) with
	corresponding variances (defined by the ALPHA field).

	See also
	MLP, MLPPRIOR, MLPPAK, MLPUNPAK


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        if isinstance(prior, netlabDistribution):
            sig = 1/sqrt(prior.index*prior.alpha)
            w = sig*randn(1, self.nwts) 
        elif isinstance(prior, types.FloatType):
            w = np.random.randn(self.nwts)*np.sqrt(1.0/prior)
        else:
            raise Exception('prior must be a scalar or a structure')

        self.unpak(w)

        
    def err(self, x, t, returnDataPrior = False):
        """MLPERR Evaluate error function for 2-layer network.

        Description
        E = MLPERR(NET, X, T) takes a network data structure NET together
        with a matrix X of input vectors and a matrix T of target vectors,
        and evaluates the error function E. The choice of error function
        corresponds to the output unit activation function. Each row of X
        corresponds to one input vector and each row of T corresponds to one
        target vector.
        
        [E, EDATA, EPRIOR] = MLPERR(NET, X, T) additionally returns the data
        and prior components of the error, assuming a zero mean Gaussian
        prior on the weights with inverse variance parameters ALPHA and BETA
        taken from the network data structure NET.
        
        See also
        MLP, MLPPAK, MLPUNPAK, MLPFWD, MLPBKP, MLPGRAD
        
        
        Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Check arguments for consistency
        errstring = self.consist('mlp', x, t)
        if errstring != None:
            raise Exception(errstring)
        y, z, a = self.fwd(x)

        if self.outfn == 'linear':        # Linear outputs
            diff = np.asarray(y - t)
            edata = 0.5*(diff*diff).sum()

        elif self.outfn == 'logistic':      # Logistic outputs
            # Ensure that log(1-y) is computable: need exp(a) > eps
            maxcut = -math.log(eps())
            # Ensure that log(y) is computable
            mincut = -math.log(1.0/realmin() - 1.0)
            a[np.nonzero(a>maxcut)] = maxcut
            a[np.nonzero(a<mincut)] = mincut
            y = 1.0/(1.0 + np.exp(-a))
            edata = - (np.multiply(t,np.log(y)) + np.multiply((1.0 - t),np.log(1.0 - y))).sum()

        elif self.outfn == 'softmax':       # Softmax outputs
            nout = a.shape[1]
            # Ensure that sum(exp(a), 2) does not overflow
            maxcut = math.log(realmax()) - math.log(nout)
            # Ensure that exp(a) > 0
            mincut = math.log(realmin())
            a[np.nonzero(a>maxcut)] = maxcut
            a[np.nonzero(a<mincut)] = mincut
            temp = np.exp(a)
            y = temp/(sum(temp, 2)*np.ones((1,nout)))
            # Ensure that log(y) is computable
            y[np.nonzero(y<realmin())] = realmin()
            edata = - (t*np.log(y)).sum()

        else:
            raise Exception('Unknown activation function ' + self.outfn)  
        return self.errbayes(edata, returnDataPrior)


    def fwd(self, x):
        """MLPFWD	Forward propagation through 2-layer network.

	Description
	Y = MLPFWD(NET, X) takes a network data structure NET together with a
	matrix X of input vectors, and forward propagates the inputs through
	the network to generate a matrix Y of output vectors. Each row of X
	corresponds to one input vector and each row of Y corresponds to one
	output vector.

	[Y, Z] = MLPFWD(NET, X) also generates a matrix Z of the hidden unit
	activations where each row corresponds to one pattern.

	[Y, Z, A] = MLPFWD(NET, X) also returns a matrix A  giving the summed
	inputs to each output unit, where each row corresponds to one
	pattern.

	See also
	MLP, MLPPAK, MLPUNPAK, MLPERR, MLPBKP, MLPGRAD


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Check arguments for consistency
        errstring = self.consist('mlp', x)
        if errstring != None:
            raise Exception(errstring)

        ndata = x.shape[0]

        z = np.tanh(np.dot(x, self.w1) + self.b1)
        a = np.dot(z, self.w2) + self.b2

        if self.outfn == 'linear':    # Linear outputs
            
            y = a

        elif self.outfn == 'logistic':  # Logistic outputs
            # Prevent overflow and underflow: use same bounds as mlperr
            # Ensure that log(1-y) is computable: need exp(a) > eps
            maxcut = -math.log(eps())
            # Ensure that log(y) is computable
            mincut = -math.log(1/realmin() - 1)
            a[np.nonzero(a>maxcut)] = maxcut
            a[np.nonzero(a<mincut)] = mincut
            y = 1/(1 + np.exp(-a))

        elif self.outfn == 'softmax':   # Softmax outputs
  
            # Prevent overflow and underflow: use same bounds as glmerr
            # Ensure that sum(exp(a), 2) does not overflow
            maxcut = math.log(realmax()) - log(self.nout)
            # Ensure that exp(a) > 0
            mincut = math.log(realmin())
            a[np.nonzero(a>maxcut)] = maxcut
            a[np.nonzero(a<mincut)] = mincut
            temp = np.exp(a)
            y = temp/np.asmatrix.sum(temp).sum(1) 

        else:
            raise Exception('Unknown activation function ' + self.outfn)
        return y, z, a

    def hdotv(self, x, t, v):
        """MLPHDOTV Evaluate the product of the data Hessian with a vector. 

	Description

	HDV = MLPHDOTV(NET, X, T, V) takes an MLP network data structure NET,
	together with the matrix X of input vectors, the matrix T of target
	vectors and an arbitrary row vector V whose length equals the number
	of parameters in the network, and returns the product of the data-
	dependent contribution to the Hessian matrix with V. The
	implementation is based on the R-propagation algorithm of
	Pearlmutter.

	See also
	MLP, MLPHESS, HESSCHEK


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Check arguments for consistency
        errstring = self.consist('mlp', x, t);
        if errstring != None:
            raise Exception(errstring)

        ndata = x.shape[0]

        y, z = mlpfwd(net, x)[:2]		# Standard forward propagation.
        zprime = (1 - z*z)			# Hidden unit first derivatives.
        zpprime = -2.0*z*zprime;		# Hidden unit second derivatives.
        vnet = self.copy()
        vnet.unpak(v)	# 		Unpack the v vector.

        # Do the R-forward propagation.

        ra1 = np.dot(x,vnet.w1) + vnet.b1
        rz = zprime*ra1
        ra2 = np.dot(rz,vnet.w2) + np.dot(z,vnet.w2) + vnet.b2

        if self.outfn == 'linear':      # Linear outputs

            ry = ra2

        elif self.outfn == 'logistic':    # Logistic outputs

            ry = y*(1 - y)*ra2

        elif self.outfn == 'softmax':     # Softmax outputs

            nout = t.shape[1]
            ry = y*ra2 - y*(np.sum(y*ra2, 1))

        else:
            raise Exception('Unknown activation function ' + net.outfn)  
        end

        # Evaluate delta for the output units.

        delout = y - t

        # Do the standard backpropagation.

        delhid = zprime*np.dot(delout,net.w2.T)

        # Now do the R-backpropagation.

        rdelhid = zpprime*ra1*np.dot(delout,net.w2) + zprime*np.dot(delout,vnet.w2.T) + zprime*np.dot(ry,net.w2.T)

        # Finally, evaluate the components of hdv and then merge into long vector.

        hw1 = np.dot(x.T,rdelhid)
        hb1 = np.sum(rdelhid, 0)
        hw2 = np.dot(z.T,ry) + np.dot(rz,delout)
        hb2 = np.sum(ry, 0)
        return np.r_[hw1.flatten(1).T, hb1.flatten(1).T, hw2.flatten(1).T, hb2.flatten(1).T]

    def hess(self, x, t, hdata):
        """MLPHESS Evaluate the Hessian matrix for a multi-layer perceptron network.

	Description
	H = MLPHESS(NET, X, T) takes an MLP network data structure NET, a
	matrix X of input values, and a matrix T of target values and returns
	the full Hessian matrix H corresponding to the second derivatives of
	the negative log posterior distribution, evaluated for the current
	weight and bias values as defined by NET.

	[H, HDATA] = MLPHESS(NET, X, T) returns both the Hessian matrix H and
	the contribution HDATA arising from the data dependent term in the
	Hessian.

	H = MLPHESS(NET, X, T, HDATA) takes a network data structure NET, a
	matrix X of input values, and a matrix T of  target values, together
	with the contribution HDATA arising from the data dependent term in
	the Hessian, and returns the full Hessian matrix H corresponding to
	the second derivatives of the negative log posterior distribution.
	This version saves computation time if HDATA has already been
	evaluated for the current weight and bias values.

	See also
	MLP, HESSCHEK, MLPHDOTV, EVIDENCE


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Check arguments for consistency
        errstring = self.consist('mlp', x, t)
        if errstring != None:
            raise Exception(errstring)

        if computeData:
            # Data term in Hessian needs to be computed
            hdata = self.datahess(x, t)

        h, hdata = self.hbayes(hdata)

        # Sub-function to compute data part of Hessian
    def datahess(self, x, t):
        
        hdata = np.zeros((self.nwts, self.nwts))

        for i in range(self.nwts):
            v = zeros(self.nwts)
            hdata[i,:] = self.hdotv(x, t, v);

        return hdata


    def pak(self):
        """MLPPAK	Combines weights and biases into one weights vector.

	Description
	W = MLPPAK(NET) takes a network data structure NET and combines the
	component weight matrices bias vectors into a single row vector W.
	The facility to switch between these two representations for the
	network parameters is useful, for example, in training a network by
	error function minimization, since a single vector of parameters can
	be handled by general-purpose optimization routines.

	The ordering of the paramters in W is defined by
	  w = [net.w1(:)', net.b1, net.w2(:)', net.b2];
	 where W1 is the first-layer weight matrix, B1 is the first-layer
	bias vector, W2 is the second-layer weight matrix, and B2 is the
	second-layer bias vector.

	See also
	MLP, MLPUNPAK, MLPFWD, MLPERR, MLPBKP, MLPGRAD


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Check arguments for consistency
        errstring = self.consist('mlp')
        if errstring != None:
            raise Exception(errstring)

        w = np.r_[self.w1.flatten(1).T, self.b1.flatten(1).T, self.w2.flatten(1).T, self.b2.flatten(1).T]
        return w


    def unpak(self, w):
        """MLPUNPAK Separates weights vector into weight and bias matrices. 
        
	Description
	NET = MLPUNPAK(NET, W) takes an mlp network data structure NET and  a
	weight vector W, and returns a network data structure identical to
	the input network, except that the first-layer weight matrix W1, the
	first-layer bias vector B1, the second-layer weight matrix W2 and the
	second-layer bias vector B2 have all been set to the corresponding
	elements of W.

	See also
	MLP, MLPPAK, MLPFWD, MLPERR, MLPBKP, MLPGRAD


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Check arguments for consistency
        errstring = self.consist('mlp')
        if errstring != None:
            raise Exception(errstring)
        
        if self.nwts != len(w):
            raise Exception('Invalid weight vector length')

        nin = self.nin
        nhidden = self.nhidden
        nout = self.nout

        mark1 = nin*nhidden
        self.w1 = np.reshape(w[0:mark1], (nin, nhidden), order='F')
        mark2 = mark1 + nhidden
        self.b1 = np.reshape(w[mark1:mark2], (1, nhidden), order='F')
        mark3 = mark2 + nhidden*nout
        self.w2 = np.reshape(w[mark2:mark3], (nhidden, nout), order='F')
        mark4 = mark3 + nout
        self.b2 = np.reshape(w[mark3:mark4], (1, nout), order='F')


    def bkp(self, x, z, deltas):
        """MLPBKP	Backpropagate gradient of error function for 2-layer network.

	Description
	G = MLPBKP(NET, X, Z, DELTAS) takes a network data structure NET
	together with a matrix X of input vectors, a matrix  Z of hidden unit
	activations, and a matrix DELTAS of the  gradient of the error
	function with respect to the values of the output units (i.e. the
	summed inputs to the output units, before the activation function is
	applied). The return value is the gradient G of the error function
	with respect to the network weights. Each row of X corresponds to one
	input vector.

	This function is provided so that the common backpropagation
	algorithm can be used by multi-layer perceptron network models to
	compute gradients for mixture density networks as well as standard
	error functions.

	See also
	MLP, MLPGRAD, MLPDERIV, MDNGRAD


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Evaluate second-layer gradients.
        gw2 = np.asarray(np.dot(z.T, deltas))
        gb2 = np.asarray(deltas.sum(0))

        # Now do the backpropagation.
        delhid = np.dot(deltas, self.w2.T)
        delhid = np.multiply(delhid,(1.0 - np.multiply(z,z)))

        # Finally, evaluate the first-layer gradients.
        gw1 = np.asarray(np.dot(x.T,delhid))
        gb1 = np.asarray(delhid.sum(0))

        g = np.r_[gw1.flatten(1).T, gb1.flatten(1).T, gw2.flatten(1).T, gb2.flatten(1).T]

        return g 


    def grad(self, x, t, returnDataPrior = False):
        """MLPGRAD Evaluate gradient of error function for 2-layer network.

	Description
	G = MLPGRAD(NET, X, T) takes a network data structure NET  together
	with a matrix X of input vectors and a matrix T of target vectors,
	and evaluates the gradient G of the error function with respect to
	the network weights. The error funcion corresponds to the choice of
	output unit activation function. Each row of X corresponds to one
	input vector and each row of T corresponds to one target vector.

	[G, GDATA, GPRIOR] = MLPGRAD(NET, X, T) also returns separately  the
	data and prior contributions to the gradient. In the case of multiple
	groups in the prior, GPRIOR is a matrix with a row for each group and
	a column for each weight parameter.

	See also
	MLP, MLPPAK, MLPUNPAK, MLPFWD, MLPERR, MLPBKP


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Check arguments for consistency
        errstring = self.consist('mlp', x, t)
        if errstring != None:
            raise Exception(errstring)
        y, z = self.fwd(x)[0:2]
        delout = y - t

        gdata = self.bkp(x, z, delout)

        return self.gbayes(gdata, returnDataPrior)


    def gbayes(self, gdata, returnDataPrior=False):
        """GBAYES	Evaluate gradient of Bayesian error function for network.

	Description
	G = GBAYES(NET, GDATA) takes a network data structure NET together
	the data contribution to the error gradient for a set of inputs and
	targets. It returns the regularised error gradient using any zero
	mean Gaussian priors on the weights defined in NET.  In addition, if
	a MASK is defined in NET, then the entries in G that correspond to
	weights with a 0 in the mask are removed.

	[G, GDATA, GPRIOR] = GBAYES(NET, GDATA) additionally returns the data
	and prior components of the error.

	See also
	ERRBAYES, GLMGRAD, MLPGRAD, RBFGRAD


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Evaluate the data contribution to the gradient.
        if self.mask != None:
            gdata = gdata(np.nonzero(self.mask))
        if self.beta != None:
            g1 = gdata*self.beta
        else:
            g1 = gdata

        # Evaluate the prior contribution to the gradient.
        if self.alpha != None:
            w = self.pak()
            if isinstance(self.alpha, types.FloatType):
                gprior = w
                g2 = self.alpha*gprior
            else:
                if self.mask != None:
                    nindx_cols = self.index.shape[1]
                    nmask_rows = np.nonzero(self.mask)[0].shape[0]
                    index = (self.index(np.nonzero(np.tile(self.mask (1, nindx_cols))))).reshape(nmask_rows, nindx_cols, order = 'F').copy()
                else:
                    index = self.index
      
                ngroups = self.alpha.shape[0]
                gprior = np.multiply(index.T,w)
                g2 = np.dot(self.alpha.T,gprior)
                
        else:
            gprior = 0
            g2 = 0


        g = g1 + g2
        if returnDataPrior:
            return g, gdata, gprior
        else:
            return g


class gmm(netlabModel):

    def __init__(self, dim, ncentres, covar_type, ppca_dim=None):
        """GMM	Creates a Gaussian mixture model with specified architecture.
    
    	Description
    	 MIX = GMM(DIM, NCENTRES, COVARTYPE) takes the dimension of the space
    	DIM, the number of centres in the mixture model and the type of the
    	mixture model, and returns a data structure MIX. The mixture model
    	type defines the covariance structure of each component  Gaussian:
    	  'spherical' = single variance parameter for each component: stored as a vector
    	  'diag' = diagonal matrix for each component: stored as rows of a matrix
    	  'full' = full matrix for each component: stored as 3d array
    	  'ppca' = probabilistic PCA: stored as principal components (in a 3d array
    	    and associated variances and off-subspace noise
    	 MIX = GMM(DIM, NCENTRES, COVARTYPE, PPCA_DIM) also sets the
    	dimension of the PPCA sub-spaces: the default value is one.
    
    	The priors are initialised to equal values summing to one, and the
    	covariances are all the identity matrix (or equivalent).  The centres
    	are initialised randomly from a zero mean unit variance Gaussian.
    	This makes use of the MATLAB function RANDN and so the seed for the
    	random weight initialisation can be set using RANDN('STATE', S) where
    	S is the state value.
    
    	The fields in MIX are
    	  
    	  type = 'gmm'
    	  nin = the dimension of the space
    	  ncentres = number of mixture components
    	  covartype = string for type of variance model
    	  priors = mixing coefficients
    	  centres = means of Gaussians: stored as rows of a matrix
    	  covars = covariances of Gaussians
    	 The additional fields for mixtures of PPCA are
    	  U = principal component subspaces
    	  lambda = in-space covariances: stored as rows of a matrix
    	 The off-subspace noise is stored in COVARS.
    
    	See also
    	GMMPAK, GMMUNPAK, GMMSAMP, GMMINIT, GMMEM, GMMACTIV, GMMPOST, 
    	GMMPROB
    

    	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        if ncentres < 1:
          raise Exception('Number of centres must be greater than zero')

        self.type = 'gmm'
        self.nin = dim
        self.ncentres = ncentres

        vartypes = ['spherical', 'diag', 'full', 'ppca']

        if covar_type not in vartypes:
            raise Exception('Undefined covariance type')
        else:
            self.covar_type = covar_type

        # Make default dimension of PPCA subspaces one.
        if covar_type == 'ppca':
            if ppca_dim == None:
                ppca_dim = 1
            if ppca_dim > dim:
                raise Exception('Dimension of PPCA subspaces must be less than data.')
            self.ppca_dim = ppca_dim

        # Initialise priors to be equal and summing to one
        self.priors = np.ones(self.ncentres) / self.ncentres

        # Initialise centres
        self.centres = np.random.randn(self.ncentres, self.nin)

        # Initialise all the variances to unity
        if self.covar_type == 'spherical':
            self.covars = np.ones((1, self.ncentres))
            self.nwts = self.ncentres + self.ncentres*self.nin + self.ncentres
        elif self.covar_type ==  'diag':
            # Store diagonals of covariance matrices as rows in a matrix
            self.covars =  np.ones((self.ncentres, self.nin))
            self.nwts = self.ncentres + self.ncentres*self.nin + self.ncentres*self.nin
        elif self.covar_type == 'full':
            # Store covariance matrices in a row vector of matrices
            self.covars = np.zeros((self.nin, self.nin, self.ncentres))
            for j in range(self.ncentres):
                self.covars[:, :, j] = np.eye(self.nin)
            self.nwts = self.ncentres + self.ncentres*self.nin + \
                self.ncentres*self.nin*self.nin
        elif self.covar_type == 'ppca':
            # This is the off-subspace noise: make it smaller than
            # lambdas
            self.covars = 0.1*np.ones(self.ncentres)
            # Also set aside storage for principal components and
            # associated variances
            init_space = np.eye(self.nin, self.ppca_dim)
            init_space[self.ppca_dim:self.nin] = np.ones((self.nin - self.ppca_dim, self.ppca_dim))
            self.U = np.zeros((init_space.shape[0], init_space.shape[1], self.ncentres))
            for j in range(self.ncentres):
                self.U[:, :, j] = init_space
            self.lambd = np.ones((self.ncentres, self.ppca_dim))
            # Take account of additional parameters
            self.nwts = self.ncentres + self.ncentres*self.nin + \
                self.ncentres + self.ncentres*self.ppca_dim + \
                self.ncentres*self.nin*self.ppca_dim
        else:
            raise Exception('Unknown covariance type ' + self.covar_type)               

    def activ(self, x):
        """GMMACTIV Computes the activations of a Gaussian mixture model.
    
    	Description
    	This function computes the activations A (i.e. the  probability
    	P(X|J) of the data conditioned on each component density)  for a
    	Gaussian mixture model.  For the PPCA model, each activation is the
    	conditional probability of X given that it is generated by the
    	component subspace. The data structure MIX defines the mixture model,
    	while the matrix X contains the data vectors.  Each row of X
    	represents a single vector.
    
    	See also
    	GMM, GMMPOST, GMMPROB
    

    	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Check that inputs are consistent:
        errstring = self.consist('gmm', x)
        if errstring != None:
            raise Exception(errstring)

        ndata = x.shape[0]
        a = np.zeros((ndata, self.ncentres))  # Preallocate matrix

        if self.covar_type == 'spherical':
            # Calculate squared norm matrix, of dimension (ndata, ncentres)
            n2 = dist2(x, self.centres)

            # Calculate width factors
            wi2 = 2*self.covars
            normal = (np.pi*wi2)**(float(self.nin)/2.0)
            
            # Now compute the activations
            a = np.exp(-(n2/wi2))/normal

        elif self.covar_type == 'diag':
            normal = (2*np.pi)**(float(self.nin)/2.0)
            s = np.prod(np.sqrt(self.covars), 1)
            for j in range(self.ncentres):
                diffs = x - self.centres[j, :]
                a[:, j] = np.exp(-0.5*np.sum((np.multiply(diffs, diffs)/self.covars[j:j+1, :]), 1))/(normal*s[j])

        elif self.covar_type == 'full':
            normal = (2*np.pi)**(float(self.nin)/2.0)
            for j in range(self.ncentres):
                diffs = x - self.centres[j, :]
                # Use Cholesky decomposition of covariance matrix to speed computation
                c = la.cholesky(self.covars[:, :, j])
                temp = la.solve(c, diffs.T).T
                a[:, j] = np.exp(-0.5*np.sum(np.multiply(temp, temp), 1))/(normal*np.prod(np.diag(c)))

        elif self.covar_type == 'ppca':
            log_normal = self.nin*math.log(2*np.pi)
            d2 = np.zeros((ndata, self.ncentres))
            logZ = np.zeros(self.ncentres)
            for i in range(self.ncentres):
                k = 1 - self.covars[i]/self.lambd[i]
                logZ[i] = log_normal + self.nin*math.log(self.covars[i]) - \
                    np.sum(np.log(1 - k))
                diffs = x - self.centres[i, :]
                proj = np.dot(diffs, self.U[:, :, i])
                d2[:,i] = (np.multiply(diffs,diffs).sum(1) - \
                    np.multiply(np.multiply(proj, k), proj).sum(1))/self.covars[i]
            a = np.exp(-0.5*(d2 + logZ))
        else:
            raise Exception('Unknown covariance type ' + self.covar_type)
        return a

    def em(self, x, options, returnFlog=False):
        """GMMEM	EM algorithm for Gaussian mixture model.

	Description
	[MIX, OPTIONS, ERRLOG] = GMMEM(MIX, X, OPTIONS) uses the Expectation
	Maximization algorithm of Dempster et al. to estimate the parameters
	of a Gaussian mixture model defined by a data structure MIX. The
	matrix X represents the data whose expectation is maximized, with
	each row corresponding to a vector.    The optional parameters have
	the following interpretations.

	OPTIONS[0] is set to 1 to display error values; also logs error
	values in the return argument ERRLOG. If OPTIONS[0] is set to 0, then
	only warning messages are displayed.  If OPTIONS[0] is -1, then
	nothing is displayed.

	OPTIONS[2] is a measure of the absolute precision required of the
	error function at the solution. If the change in log likelihood
	between two steps of the EM algorithm is less than this value, then
	the function terminates.

	OPTIONS[4] is set to 1 if a covariance matrix is reset to its
	original value when any of its singular values are too small (less
	than MIN_COVAR which has the value eps).   With the default value of
	0 no action is taken.

	OPTIONS[13] is the maximum number of iterations; default 100.

	The optional return value OPTIONS contains the final error value
	(i.e. data log likelihood) in OPTIONS[7].

	See also
	GMM, GMMINIT


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Check that inputs are consistent
        errstring = self.consist('gmm', x)
        if errstring != None:
            raise Exception(errstring)
        

        ndata, xdim = x.shape

        # Sort out the options
        if options[13]:
            niters = options[13]
        else:
            niters = 100

        display = options[0]
        store = False
        if returnFlog:
            store = True	# Store the error values to return them
            errlog = np.zeros(niters)
        test = False
        if options[2] > 0.0:
            test = True	# Test log likelihood for termination

        check_covars = 0
        if options[4] >= 1:
            if display >= 0:
                print 'check_covars is on'
            check_covars = True	# Ensure that covariances don't collapse
            MIN_COVAR = eps()	# Minimum singular value of covariance matrix
            init_covars = self.covars

        # Main loop of algorithm
        for n in range(niters):
  
            # Calculate posteriors based on old parameters
            post, act = self.post(x)
  
            # Calculate error value if needed
            if display or store or test:
                prob = np.dot(act, self.priors)
                # Error value is negative log likelihood of data
                e = - np.sum(np.log(prob))
                if store:
                    errlog[n] = e
                if display > 0:
                    print 'Cycle ', n, ' Error ', e
                if test:
                    if n > 0 and abs(e - eold) < options[2]:
                        options[7] = e
                        if returnFlog:
                            return errlog
                        else:
                            return
                    else:
                        eold = e
                    
    
                
  
            # Adjust the new estimates for the parameters
            new_pr = np.sum(post, 0)
            new_c = np.dot(post.T,x)
  
            # Now move new estimates to old parameter vectors
            self.priors = new_pr/ndata
  
            self.centres = new_c/new_pr.reshape(self.ncentres, 1)
  
            if self.covar_type == 'spherical':
                v = np.zeros(self.ncentres)
                n2 = dist2(x, self.centres)
                for j in range(self.ncentres):
                    v[j] = np.dot(post[:,j].T, n2[:,j])
                self.covars = ((v/new_pr))/self.nin;
                if check_covars:
                    # Ensure that no covariance is too small
                    for j in range(self.ncentres):
                        if self.covars[j] < MIN_COVAR:
                            self.covars[j] = init_covars[j]
            elif self.covar_type == 'diag':
                for j in range(self.ncentres):
                    diffs = x - self.centres[j,:]
                    self.covars[j,:] = np.sum(np.multiply(np.multiply(diffs, diffs), post[:,j:j+1]), 0)/new_pr[j]
                if check_covars:
                    # Ensure that no covariance is too small
                    for j in range(self.ncentres):
                        if np.min(self.covars[j,:]) < MIN_COVAR:
                            self.covars[j,:] = init_covars[j,:]
            elif self.covar_type == 'full':
                for j in range(self.ncentres):
                    diffs = x - self.centres[j,:];
                    diffs = np.multiply(diffs, np.sqrt(post[:,j:j+1]))
                    self.covars[:,:,j] = np.dot(diffs.T,diffs)/new_pr[j]
                if check_covars:
                    # Ensure that no covariance is too small
                    for j in range(self.ncentres):
                        if np.min(la.svd(self.covars[:,:,j], compute_uv=False)) < MIN_COVAR:
                            self.covars[:,:,j] = init_covars[:,:,j]
            elif self.covar_type == 'ppca':
                for j in range(self.ncentres):
                    diffs = x - self.centres[j,:]
                    diffs = np.multiply(diffs,np.sqrt(post[:,j:j+1]))
                    tempcovars, tempU, templambda = ppca(np.dot(diffs.T,diffs)/new_pr[j], self.ppca_dim)
                    if len(templambda) != self.ppca_dim:
                        raise Exception('Unable to extract enough components')
                    else: 
                        self.covars[j] = tempcovars
                        self.U[:, :, j] = tempU
                        self.lambd[j, :] = templambda
                        
                    if check_covars:
                        if self.covars[j] < MIN_COVAR:
                            self.covars[j] = init_covars[j]
            else:
                raise Exception('Unknown covariance type ' + self.covar_type)

        options[7] = -np.sum(np.log(self.prob(x)))
        if display >= 0:
            print maxitmess()
        if returnFlog:
            return errlog
        else:
            return

    def init(self, x, options):
        """GMMINIT Initialises Gaussian mixture model from data

        Description
        MIX = GMMINIT(MIX, X, OPTIONS) uses a dataset X to initialise the
        parameters of a Gaussian mixture model defined by the data structure
        MIX.  The k-means algorithm is used to determine the centres. The
        priors are computed from the proportion of examples belonging to each
        cluster. The covariance matrices are calculated as the sample
        covariance of the points associated with (i.e. closest to) the
        corresponding centres. For a mixture of PPCA model, the PPCA
        decomposition is calculated for the points closest to a given centre.
        This initialisation can be used as the starting point for training
        the model using the EM algorithm.

        See also
        GMM


        Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        ndata, xdim = x.shape

        # Check that inputs are consistent
        errstring = self.consist('gmm', x)
        if errstring != None:
            raise Exception(errstring)

        # Arbitrary width used if variance collapses to zero: make it 'large' so
        # that centre is responsible for a reasonable number of points.
        GMM_WIDTH = 1.0

        # Use kmeans algorithm to set centres
        options[4] = 1	
        self.centres, post, errlog = kmeans(self.centres, x, options)

        # Set priors depending on number of points in each cluster
        cluster_sizes = post.sum(0)  # Make sure that no prior is zero
        cluster_sizes[np.nonzero(cluster_sizes<1)]=1.0
        self.priors = cluster_sizes/sum(cluster_sizes) # Normalise priors

        if self.covar_type=='spherical':
            if self.ncentres > 1:
                # Determine widths as distance to nearest centre 
                # (or a constant if this is zero)
                cdist = dist2(self.centres, self.centres)
                cdist = cdist + np.diag(np.ones(self.ncentres)*realmax())
                self.covars = cdist.min(0)
                self.covars = self.covars + GMM_WIDTH*(self.covars < eps)
            else:
                self.covars = np.diag(cov(x, rowvar=1)).mean()
        elif self.covar_type=='diag':
            for j in range(self.ncentres):
                # Pick out data points belonging to this centre
                c = x[np.nonzero(post[:, j])[0],:]
                diffs = c - self.centres[j, :]
                self.covars[j, :] = np.multiply(diffs,diffs).sum(0)/c.shape[0]
                # Replace small entries by GMM_WIDTH value
                self.covars[j, :] = self.covars[j, :] + GMM_WIDTH*(self.covars[j, :]<eps())
        elif self.covar_type=='full':
            for j in range(self.ncentres):
                # Pick out data points belonging to this centre
                c = x[np.nonzero(post[:, j])[0],:]
                diffs = c - self.centres[j, :]
                self.covars[:,:,j] = np.dot(diffs.T,diffs)/c.shape[0]
                # Add GMM_WIDTH*Identity to rank-deficient covariance matrices
                if mrank(self.covars[:,:,j]) < self.nin:
                    self.covars[:,:,j] = self.covars[:,:,j] + GMM_WIDTH*eye(self.nin)


        elif self.covar_type=='ppca':
            for j in range(self.ncentres):
                # Pick out data points belonging to this centre
                c = x[np.nonzero(post[:, j])[0],:]
                diffs = c - self.centres[j, :]
                tempcovars, tempU, templambda = ppca(np.dot(diffs.T, diffs)/c.shape[0], self.ppca_dim)
                if len(templambda) != self.ppca_dim:
                    raise Exception('Unable to extract enough components')
                else:
                    self.covars[j] = tempcovars
                    self.U[:, :, j] = tempU
                    self.lambd[j, :] = templambda
        else:
            raise Exception('Unknown covariance type ' + self.covar_type)

    def post(self, x):
        """GMMPOST Computes the class posterior probabilities of a Gaussian mixture model.
        
        Description
	This function computes the posteriors POST (i.e. the probability of
	each component conditioned on the data P(J|X)) for a Gaussian mixture
	model.   The data structure MIX defines the mixture model, while the
	matrix X contains the data vectors.  Each row of X represents a
	single vector.

	See also
	GMM, GMMACTIV, GMMPROB

        
	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Check that inputs are consistent
        errstring = self.consist('gmm', x)
        if errstring != None:
            raise Exception(errstring)

        ndata = x.shape[0]

        a = self.activ(x)

        post = np.multiply(self.priors,a)
        s = np.sum(post, 1)
        if np.any(s==0):
            print 'Warning: Some zero posterior probabilities'
            # Set any zeros to one before dividing
            zero_rows = np.nonzeros(s==0)[0]
            s = s + (s==0)
            post[zero_rows] = 1/self.ncentres


        post = post/np.reshape(s, (ndata, 1))
        return post, a

    def prob(self, x):
        """GMMPROB Computes the data probability for a Gaussian mixture model.
        
        Description
        This function computes the unconditional data density P(X) for a
        Gaussian mixture model.  The data structure MIX defines the mixture
        model, while the matrix X contains the data vectors.  Each row of X
        represents a single vector.
        
        See also
        GMM, GMMPOST, GMMACTIV
        
        
        Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""
        
        # Check that inputs are consistent
        errstring = self.consist('gmm', x)
        if errstring != None:
            raise Exception(errstring)

        # Compute activations
        a = self.activ(x)

        # Form dot product with priors
        return np.dot(a, self.priors)


    def samp(self, n, returnLabels=False):
        """GMMSAMP Sample from a Gaussian mixture distribution.

	Description

	DATA = GSAMP(MIX, N) generates a sample of size N from a Gaussian
	mixture distribution defined by the MIX data structure. The matrix X
	has N rows in which each row represents a MIX.NIN-dimensional sample
	vector.

	[DATA, LABEL] = GMMSAMP(MIX, N) also returns a column vector of
	classes (as an index 1..N) LABEL.

	See also
	GSAMP, GMM


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Check input arguments
        errstring = self.consist('gmm')
        if errstring != None:
            raise Exception(errstring)
        if n < 1:
            raise Exception('Number of data points must be positive')

        # Determine number to sample from each component
        priors = np.random.rand(n)

        # Pre-allocate data array
        data = np.zeros((n, self.nin))
        if returnLabels:
            label = np.zeros((n, 1))
        cum_prior = 0		# Cumulative sum of priors
        total_samples = 0	# Cumulative sum of number of sampled points
        for j in range(self.ncentres):
            num_samples = np.sum(np.logical_and(priors >= cum_prior, priors < cum_prior + self.priors[j]))
            # Form a full covariance matrix
            if self.covar_type == 'spherical':
                covar = self.covars[j] * np.eye(self.nin)
            elif self.covar_type == 'diag':
                covar = diag(self.covars[j, :])
            elif self.covar_type == 'full':
                covar = self.covars[:, :, j]
            elif self.covar_type == 'ppca':
                covar = self.covars[j] * np.eye(self.nin) \
                    + np.dot(np.dot(self.U[:, :, j], (np.diag(self.lambd[j, :])-(self.covars[j]*np.eye(self.ppca_dim)))), self.U[:, :, j].T)
            else:
                raise Exception('Unknown covariance type ' + self.covar_type)
            data[total_samples:total_samples+num_samples] = gsamp(self.centres[j, :], covar, num_samples)
            if returnLabels:
                label[total_samples:total_samples+num_samples] = j
            cum_prior = cum_prior + self.priors[j]
            total_samples = total_samples + num_samples
        if returnLabels:
            return data, label
        else:
            return data
        


    def unpak(self, p):
        """GMMUNPAK Separates a vector of Gaussian mixture model parameters into its components.

        Description
        MIX = GMMUNPAK(MIX, P) takes a GMM data structure MIX and  a single
        row vector of parameters P and returns a mixture data structure
        identical to the input MIX, except that the mixing coefficients
        PRIORS, centres CENTRES and covariances COVARS  (and, for PPCA, the
        lambdas and U (PCA sub-spaces)) are all set to the corresponding
        elements of P.

        See also
        GMM, GMMPAK


        Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        errstring = self.consist('gmm')
        if errstring != None:
            raise Exception(errstring)
        if self.nwts != len(p):
            raise Exception('Invalid weight vector length')

        mark1 = self.ncentres
        mark2 = mark1 + self.ncentres*self.nin

        self.priors = p[0:mark1]
        self.centres = p[mark1:mark2].reshape(self.ncentres, self.nin, order='F')
        if self.covar_type == 'spherical':
            mark3 = self.ncentres*(2 + self.nin)
            self.covars = p[mark2:mark3].reshape(1, self.ncentres, order='F')
        elif self.covar_type == 'diag':
            mark3 = self.ncentres*(1 + self.nin + self.nin)
            self.covars = p[mark2:mark3].reshape(self.ncentres, self.nin, order='F')
        elif self.covar_type == 'full':
            mark3 = self.ncentres*(1 + self.nin + self.nin*self.nin)
            self.covars = p[mark2:mark3].reshape(self.nin, self.nin, self.ncentres, order='F')
        elif self.covar_type == 'ppca':
            mark3 = self.ncentres*(2 + self.nin)
            self.covars = p[mark2:mark3]
            # Now also extract k and eigenspaces
            mark4 = mark3 + self.ncentres*self.ppca_dim
            self.lambd = p[mark3:mark4].reshape(self.ncentres, self.ppca_dim, order='F')
            self.U = p[mark4 + 1:-1].reshape(self.nin, self.ppca_dim, self.ncentres, order='F')
        else:
            raise Exception('Unknown covariance type ' + self.covar_type)

    def pak(self):
        """GMMPAK	Combines all the parameters in a Gaussian mixture model into one vector.

        Description
        P = GMMPAK(NET) takes a mixture data structure MIX  and combines the
        component parameter matrices into a single row vector P.

        See also
        GMM, GMMUNPAK


        Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        errstring = self.consist('gmm')
        if errstring != None:
            raise Exception(errstring)

        p = np.r_[self.priors.flatten(1).T, self.centres.flatten(1).T, self.covars.flatten(1).T]
        if self.covar_type == 'ppca':
            p = np.r_[p, self.lambd.flatten(1).T, self.U.flatten(1).T]
        return p


class som(netlabModel):

    def __init__(self, nin, map_size):

        """SOM	Creates a Self-Organising Map.

	Description
	NET = SOM(NIN, MAP_SIZE) creates a SOM NET with input dimension (i.e.
	data dimension) NIN and map dimensions MAP_SIZE.  Only two-
	dimensional maps are currently implemented.

	The fields in NET are
	  type = 'som'
	  nin = number of inputs
	  map_dim = dimension of map (constrained to be 2)
	  map_size = grid size: number of nodes in each dimension
	  num_nodes = number of nodes: the product of values in map_size
	  map = map_dim+1 dimensional array containing nodes
	  inode_dist = map of inter-node distances using Manhatten metric

	The map contains the node vectors arranged column-wise in the first
	dimension of the array.

	See also
	KMEANS, SOMFWD, SOMTRAIN


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        self.type = 'som'
        self.nin = nin

        # Create Map of nodes
        if np.any(np.round(map_size) != map_size) or np.any(map_size < 1):
            raise Exception('SOM specification must contain positive integers')

        self.map_dim = len(map_size)
        if self.map_dim != 2:
            raise Exception('SOM is a 2 dimensional map')
        
        self.num_nodes = np.prod(map_size)
        # Centres are stored by column as first index of multi-dimensional array.
        # This makes extracting them later more easy.
        # Initialise with rand to create square grid
        self.map = np.random.rand(nin, map_size[0], map_size[1])
        self.map_size = map_size

        # Crude function to compute inter-node distances
        self.inode_dist = np.zeros((map_size[0], map_size[1], self.num_nodes))
        for m in range(self.num_nodes):
            node_loc = np.array([np.fix(m/map_size[1]), m%map_size[1]])
            for k in range(map_size[0]):
                for l in range(map_size[1]):
                    self.inode_dist[k, l, m] = round(np.max(np.abs(np.array([k, l]) - node_loc)))


    def fwd(self, x):
        """SOMFWD	Forward propagation through a Self-Organising Map.

	Description
	D2 = SOMFWD(NET, X) propagates the data matrix X through  a SOM NET,
	returning the squared distance matrix D2 with dimension NIN by
	NUM_NODES.  The $i$th row represents the squared Euclidean distance
	to each of the nodes of the SOM.

	[D2, WIN_NODES] = SOMFWD(NET, X) also returns the indices of the
	winning nodes for each pattern.

	See also
	SOM, SOMTRAIN


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Check for consistency
        errstring = self.consist('som', x)
        if errstring != None:
            raise Exception(errstring)

        # Turn nodes into matrix of centres
        nodes = np.reshape(self.map, (self.nin, self.num_nodes), order='F').T
        # Compute squared distance matrix
        d2 = dist2(x, nodes)
        # Find winning node for each pattern: minimum value in each row
        win_nodes = np.argmin(d2, 1)
        w = np.min(d2, 1)
        return d2, win_nodes 


    def pak(self):
        """SOMPAK	Combines node weights into one weights matrix.

	Description
	C = SOMPAK(NET) takes a SOM data structure NET and combines the node
	weights into a matrix of centres C where each row represents the node
	vector.

	The ordering of the parameters in W is defined by the indexing of the
	multi-dimensional array NET.MAP.

	See also
	SOM, SOMUNPAK


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        errstring = self.consist('som')
        if errstring != None:
            raise Exception(errstring)
        # Returns map as a sequence of row vectors
        c = np.reshape(self.map, (self.nin, self.num_nodes), order='F').T
        return c 

    def unpak(self, w):
        """SOMUNPAK Replaces node weights in SOM.

	Description
	NET = SOMUNPAK(NET, W) takes a SOM data structure NET and weight
	matrix W (each node represented by a row) and puts the nodes back
	into the multi-dimensional array NET.MAP.

	The ordering of the parameters in W is defined by the indexing of the
	multi-dimensional array NET.MAP.

	See also
	SOM, SOMPAK


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        errstring = self.consist('som')
        if errstring != None:
            raise Exception(errstring)
        
        # Put weights back into network data structure
        self.map = np.reshape(w.T, (self.nin, self.map_size[0], self.map_size[1]), order='F')


    def train(self, options, x):
        """SOMTRAIN Kohonen training algorithm for SOM.

	Description
	NET = SOMTRAIN{NET, OPTIONS, X) uses Kohonen's algorithm to train a
	SOM.  Both on-line and batch algorithms are implemented. The learning
	rate (for on-line) and neighbourhood size decay linearly. There is no
	error function minimised during training (so there is no termination
	criterion other than the number of epochs), but the  sum-of-squares
	is computed and returned in OPTIONS[7].

	The optional parameters have the following interpretations.

	OPTIONS[0] is set to 1 to display error values; also logs learning
	rate ALPHA and neighbourhood size NSIZE. Otherwise nothing is
	displayed.

	OPTIONS[4] determines whether the patterns are sampled randomly with
	replacement. If it is 0 (the default), then patterns are sampled in
	order.  This is only relevant to the on-line algorithm.

	OPTIONS[5] determines if the on-line or batch algorithm is used. If
	it is 1 then the batch algorithm is used.  If it is 0 (the default)
	then the on-line algorithm is used.

	OPTIONS[13] is the maximum number of iterations (passes through the
	complete pattern set); default 100.

	OPTIONS[14] is the final neighbourhood size; default value is the
	same as the initial neighbourhood size.

	OPTIONS[15] is the final learning rate; default value is the same as
	the initial learning rate.

	OPTIONS[16] is the initial neighbourhood size; default 0.5*maximum
	map size.

	OPTIONS[17] is the initial learning rate; default 0.9.  This
	parameter must be positive.

	See also
	KMEANS, SOM, SOMFWD


	Copyright (c) Ian T Nabney (1996-2001)
        and Neil D. Lawrence (2009) (translation to python)"""

        # Check arguments for consistency
        errstring = self.consist('som', x)
        if errstring != None:
            raise Exception(errstring)

        # Set number of iterations in convergence phase
        if not options[13]:
            options[13] = 100

        niters = options[13]

        # Learning rate must be positive
        if (options[17] > 0):
            alpha_first = options[17]
        else:
            alpha_first = 0.9

        # Final learning rate must be no greater than initial learning rate
        if (options[15] > alpha_first or options[15] < 0):
            alpha_last = alpha_first
        else:
            alpha_last = options[15]

        # Neighbourhood size
        if (options[16] >= 0):
            nsize_first = options[16]
        else:
            nsize_first = np.max(self.map_dim)/2

        # Final neighbourhood size must be no greater than initial size
        if (options[14] > nsize_first or options[14] < 0):
            nsize_last = nsize_first
        else:
            nsize_last = options[14]

        ndata = x.shape[0]

        if options[5]:
            # Batch algorithm
            H = np.zeros((ndata, self.num_nodes))

        # Put weights into matrix form
        tempw = self.pak()

        # Then carry out training
        j = 0
        while j < niters:
            if options[5]:
                # Batch version of algorithm
                alpha = 0.0
                frac_done = float(niters - j)/float(niters)
                # Compute neighbourhood
                nsize = round((nsize_first - nsize_last)*frac_done + nsize_last)

                # Find winning node: put weights back into net so that we can
                # call somunpak
                self.unpak(tempw)
                temp, bnode = self.fwd(x)
                for k in range(ndata):
                    H[k, :] = np.reshape(self.inode_dist[:, :, bnode[k]]<=nsize, (1, self.num_nodes))
                s = np.sum(H, 0)
                for k in range(self.num_nodes):
                    if s[k] > 0:
                        tempw[k, :] = np.sum(H[:, k]*x.T, 1)/s[k]
            else:
                # On-line version of algorithm
                if options[4]:
                    # Randomise order of pattern presentation: with replacement
                    pnum = np.floor(np.random.rand(ndata)*ndata)
                else:
                    pnum = np.r_[0:ndata]
                # Cycle through dataset
                for k in range(ndata):
                    # Fraction done
                    frac_done = (((niters+1)*ndata)-((j+1)*ndata + (k+1)))/((niters+1)*ndata);
                    # Compute learning rate
                    alpha = (alpha_first - alpha_last)*frac_done + alpha_last
                    # Compute neighbourhood
                    nsize = round((nsize_first - nsize_last)*frac_done + nsize_last)
                    # Find best node
                    pat_diff = x[pnum[k], :] - tempw
                    bnode = np.argmin(np.sum(np.abs(pat_diff),1))

                    # Now update neighbourhood
                    neighbourhood = (self.inode_dist[:, :, bnode] <= nsize)
                    tempw = tempw + alpha*(neighbourhood.reshape((-1, 1))*pat_diff)
            j = j + 1
            if options[0]:
                # Print iteration information
                print 'Iteration ', j, '; alpha = ', alpha, ', nsize = ', nsize, '. ',

                # Print sum squared error to nearest node
                d2 = dist2(tempw, x)
                print 'Error = ', np.min(d2, 0).sum()


        self.unpak(tempw)
        options[7] = np.min(dist2(tempw, x), 0).sum()
