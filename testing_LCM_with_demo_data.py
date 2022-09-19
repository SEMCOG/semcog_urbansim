import pandas as pd
import numpy as np

from dcm_ard_libs import minimize, neglog_DCM

theta = pd.read_csv('~/DCM-ARD/theta.csv', header=None).values
# theta: m x 2
# X: n x m
X1 = pd.read_csv('~/DCM-ARD/X.csv', header=None).values
X2 = pd.read_csv('~/DCM-ARD/X2.csv', header=None).values
X3 = pd.read_csv('~/DCM-ARD/X3.csv', header=None).values
X = {
	0: X1,
	1: X2,
	2: X3
}
theta1 = np.zeros((X[0].shape[1],1))
theta2 = np.zeros((X[1].shape[1],1))
theta3 = np.zeros((X[2].shape[1],1))
theta = {
	0: theta1,
	1: theta2,
	2: theta3
}
# Y: n x 1
Y = pd.read_csv('~/DCM-ARD/Y.csv', header=None).values
Y = Y-1 # changed to 0 indexed from 1 indexed
# Y = (Y == 1).astype(int)[:,[0]]
# onehot: n x 2
Y_onehot = pd.read_csv('~/DCM-ARD/Y_onehot.csv', header=None).values
# Y_onehot = Y_onehot[:, [0,1]]
# Y_onehot[:, 1] = (Y_onehot[:, 0]==0).astype(int)

availableChoices = pd.read_csv('~/DCM-ARD/availableChoices.csv', header=None).values
# availableChoices = availableChoices[:, [0,1]]

theta_optim_full = minimize( theta, neglog_DCM, -20000, X, Y, Y_onehot, availableChoices)

print('done')