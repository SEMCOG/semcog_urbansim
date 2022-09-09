import numpy as np
import pandas as pd
    
def minimize(X = None, f = None, length = None, *args): 
    # Minimize a differentiable multivariate function using conjugate gradients.
    
    # Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, ... )
    
    # X       initial guess; may be of any type, including struct and cell array
# f       the name or pointer to the function to be minimized. The function
#         f must return two arguments, the value of the function, and it's
#         partial derivatives wrt the elements of X. The partial derivative
#         must have the same type as X.
# length  length of the run; if it is positive, it gives the maximum number of
#         line searches, if negative its absolute gives the maximum allowed
#         number of function evaluations. Optionally, length can have a second
#         component, which will indicate the reduction in function value to be
#         expected in the first line-search (defaults to 1.0).
# P1, P2, ... parameters are passed to the function f.
    
    # X       the returned solution
# fX      vector of function values indicating progress made
# i       number of iterations (line searches or function evaluations,
#         depending on the sign of "length") used at termination.
    
    # The function returns when either its length is up, or if no further progress
# can be made (ie, we are at a (local) minimum, or so close that due to
# numerical problems, we cannot get any closer). NOTE: If the function
# terminates within a few iterations, it could be an indication that the
# function values and derivatives are not consistent (ie, there may be a bug in
# the implementation of your "f" function).
    
    # The Polack-Ribiere flavour of conjugate gradients is used to compute search
# directions, and a line search using quadratic and cubic polynomial
# approximations and the Wolfe-Powell stopping criteria is used together with
# the slope ratio method for guessing initial step sizes. Additionally a bunch
# of checks are made to make sure that exploration is taking place and that
# extrapolation will not be unboundedly large.
    
    # See also: checkgrad
    
    # Copyright (C) 2001 - 2010 by Carl Edward Rasmussen, 2010-01-03
    
    INT = 0.1
    
    EXT = 3.0
    
    MAX = 20
    
    RATIO = 10
    
    SIG = 0.1
    RHO = SIG / 2
    
    # Powell conditions. SIG is the maximum allowed absolute ratio between
# previous and new slopes (derivatives in the search direction), thus setting
# SIG to low (positive) values forces higher precision in the line-searches.
# RHO is the minimum allowed fraction of the expected (from the slope at the
# initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
# Tuning of SIG (depending on the nature of the function to be optimized) may
# speed up the minimization; it is probably not worth playing much with RHO.
    
    # The code falls naturally into 3 parts, after the initial line search is
# started in the direction of steepest descent. 1) we first enter a while loop
# which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
# have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
# enter the second loop which takes p2, p3 and p4 chooses the subinterval
# containing a (local) minimum, and interpolates it, unil an acceptable point
# is found (Wolfe-Powell conditions). Note, that points are always maintained
# in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
# conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
# was a problem in the previous line-search. Return the best value so far, if
# two consecutive line-searches fail, or whenever we run out of function
# evaluations or line-searches. During extrapolation, the "f" function may fail
# either with an error or returning Nan or Inf, and minimize should handle this
# gracefully.
    
    if np.amax(length.shape) == 2:
        red = len(2)
        length = len(1)
    else:
        red = 1
    
    if length > 0:
        S = 'Linesearch'
    else:
        S = 'Function evaluation'
    
    i = 0
    
    ls_failed = 0
    
    # f0,df0 = feval(f,X,varargin[:])
    f0,df0 = f(X,args[:])
    
    Z = X
    X = unwrap(X)
    df0 = unwrap(df0)
    print('%s %6i;  Value %4.6e\r' % (S,i,f0))
    #if exist('fflush','builtin') fflush(stdout); end
    fX = f0
    i = i + (length < 0)
    
    s = - df0
    d0 = - np.transpose(s) * s
    
    x3 = red / (1 - d0)
    
    while i < np.abs(length):

        i = i + (length > 0)
        X0 = X
        F0 = f0
        dF0 = df0
        if length > 0:
            M = MAX
        else:
            M = np.amin(MAX,- length - i)
        while 1:

            x2 = 0
            f2 = f0
            d2 = d0
            f3 = f0
            df3 = df0
            success = 0
            while not success  and M > 0:

                try:
                    M = M - 1
                    i = i + (length < 0)
                    # f3,df3 = feval(f,rewrap(Z,X + x3 * s),args[:])
                    f3,df3 = f(rewrap(Z,X + x3 * s),args[:])
                    df3 = unwrap(df3)
                    if np.isnan(f3) or np.isinf(f3) or np.any(np.isnan(df3) + np.isinf(df3)):
                        raise Exception(' ')
                    success = 1
                finally:
                    pass

            if f3 < F0:
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3
            d3 = np.transpose(df3) * s
            if d3 > SIG * d0 or f3 > f0 + x3 * RHO * d0 or M == 0:
                break
            x1 = x2
            f1 = f2
            d1 = d2
            x2 = x3
            f2 = f3
            d2 = d3
            A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
            B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
            x3 = x1 - d1 * (x2 - x1) ** 2 / (B + np.sqrt(B * B - A * d1 * (x2 - x1)))
            if not True  or np.isnan(x3) or np.isinf(x3) or x3 < 0:
                x3 = x2 * EXT
            else:
                if x3 > x2 * EXT:
                    x3 = x2 * EXT
                else:
                    if x3 < x2 + INT * (x2 - x1):
                        x3 = x2 + INT * (x2 - x1)

        while (np.abs(d3) > - SIG * d0 or f3 > f0 + x3 * RHO * d0) and M > 0:

            if d3 > 0 or f3 > f0 + x3 * RHO * d0:
                x4 = x3
                f4 = f3
                d4 = d3
            else:
                x2 = x3
                f2 = f3
                d2 = d3
            if f4 > f0:
                x3 = x2 - (0.5 * d2 * (x4 - x2) ** 2) / (f4 - f2 - d2 * (x4 - x2))
            else:
                A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)
                B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                x3 = x2 + (np.sqrt(B * B - A * d2 * (x4 - x2) ** 2) - B) / A
            if np.isnan(x3) or np.isinf(x3):
                x3 = (x2 + x4) / 2
            x3 = np.amax(np.amin(x3,x4 - INT * (x4 - x2)),x2 + INT * (x4 - x2))
            # f3,df3 = feval(f,rewrap(Z,X + x3 * s),args[:])
            f3,df3 = f(rewrap(Z,X + x3 * s), args[:])
            df3 = unwrap(df3)
            if f3 < F0:
                X0 = X + x3 * s
                F0 = f3
                dF0 = df3
            M = M - 1
            i = i + (length < 0)
            d3 = np.transpose(df3) * s

        if np.abs(d3) < - SIG * d0 and f3 < f0 + x3 * RHO * d0:
            X = X + x3 * s
            f0 = f3
            fX = np.transpose(np.array([np.transpose(fX),f0]))
            print('%s %6i;  Value %4.6e\r' % (S,i,f0))
            #    if exist('fflush','builtin') fflush(stdout); end
            s = (np.transpose(df3) * df3 - np.transpose(df0) * df3) / (np.transpose(df0) * df0) * s - df3
            df0 = df3
            d3 = d0
            d0 = np.transpose(df0) * s
            if d0 > 0:
                s = - df0
                d0 = - np.transpose(s) * s
            x3 = x3 * np.amin(RATIO,d3 / (d0 - np.finfo(float).tiny))
            ls_failed = 0
        else:
            X = X0
            f0 = F0
            df0 = dF0
            if ls_failed or i > np.abs(length):
                break
            s = - df0
            d0 = - np.transpose(s) * s
            x3 = 1 / (1 - d0)
            ls_failed = 1

    
    X = rewrap(Z,X)
    # fprintf('\n'); if exist('fflush','builtin') fflush(stdout); end
    return X, fX, i
    
    
def unwrap(s = None): 
    # Extract the numerical values from "s" into the column vector "v". The
# variable "s" can be of any type, including struct and cell array.
# Non-numerical elements are ignored. See also the reverse rewrap.m.
    v = []
    if pd.api.types.is_numeric_dtype(s):
        v = s
    else:
        for i in np.arange(1,np.asarray(s).size+1).reshape(-1):
            v = np.array([[v],[unwrap(s[i])]])
    return v
    
    
def rewrap(s = None,v = None): 
    # Map the numerical elements in the vector "v" onto the variables "s" which can
# be of any type. The number of numerical elements must match; on exit "v"
# should be empty. Non-numerical entries are just copied. See also unwrap.m.
    if pd.api.types.is_numeric_dtype(s):
        if v.size < s.size:
            raise Exception('The vector for conversion contains too few elements')
        # s = np.reshape(v(np.arange(1,np.asarray(s).size+1)), tuple(s.shape), order="F")
        s = v[:s.size].reshape(s.size)
        # v = v(np.arange(np.asarray(s).size + 1,end()+1))
        v = v[s.size:]
    else:
        for i in np.arange(np.asarray(s).size).reshape(-1):
            s[i],v = rewrap(s[i],v)
    return s, v

def neglog_DCM(theta = None,X = None,Y = None,T = None,availableChoices = None): 
    #function [g, dg] = neglog_DCM(theta, X, Y, T, availableChoices)
    
    # (C) Filipe Rodrigues (2019)
    
    N,K = T.shape
    F = np.zeros((N,K))
    for k in np.arange(K):
        F[:,k] = X[k] * theta[k]
    
    ma = np.amax(F, 1)
    Fma = F - ma
    expF = np.exp(Fma)
    expF = np.multiply(expF,availableChoices)
    
    normExpF = np.sum(expF, 1)
    S = expF / normExpF
    # Yind = sub2ind(Fma.shape,np.transpose(np.array([np.arange(N)])),Y)
    # g = - sum(Fma(Yind) - np.log(normExpF))
    Yind = np.ravel_multi_index( [np.arange(N), Y], Fma.shape)
    g = -np.sum(Fma[Yind] - np.log(normExpF));
    # dg = cell(3,1)
    dg = np.empty([3,1])
    for k in np.arange(K):
        dg[k] = - X[k].T * (T[:,k] - S[:,k])
    
    return g,dg