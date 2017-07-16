#coding=utf-8

import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from os import environ
from os.path import dirname
from os.path import join
from os.path import exists
import codecs


def load_data_from_txt():
	"""data file form spark mllib example
    -9.490009878824548 1:0.4551273600657362 2:0.36644694351969087 3:-0.38256108933468047 4:-0.4458430198517267 5:0.33109790358914726 6:0.8067445293443565 7:-0.2624341731773887 8:-0.44850386111659524 9:-0.07269284838169332 10:0.5658035575800715
	each line has 11 fields and the last ten fields are ten features of the data. 
	"""
	file_path = dirname(__file__)
	datafile =  join(file_path, "data","sample_linear_regression_data.txt")
	data = np.empty((501, 10))
	target = np.empty((501,))

	i = 0
	with open(datafile) as f:
	  for line in f:
	      fields = line.strip().split(' ')
	      features = [item[item.find(':')+1:] for item in fields[1:]]   
	      target[i] =  np.asarray(fields[0], dtype=np.float64)
	      data[i] = np.asarray(features, dtype=np.float64)
	      i += 1
	      if i>500: break
	######In order to do multiple regression we need to add a column of 1s for x0
	x = np.array([np.concatenate((v,[1])) for v in data])
	return x, target


def verify_data(data,target):
	print(data.shape)
	print(target.shape)
	np.set_printoptions(precision=2, linewidth=120, suppress=True, edgeitems=4)
	print(data)

def plot_result(pred,real):
	#%matplotlib inline
	plt.plot(pred, real,'ro')
	plt.plot([0,6],[0,6], 'g-')
	plt.xlabel('predicted')
	plt.ylabel('real')
	plt.show()

def printError(p,y):
	# Now we can constuct a vector of errors
	err = abs(p-y)
	# Let's see the error on the first 10 predictions
	print("first ten errors\n", err[:10])
	# Dot product of error vector with itself gives us the sum of squared errors
	total_error = np.dot(err,err)
	# Compute RMSE
	rmse_train = np.sqrt(total_error/len(p))
	print ("RMSE:\n", rmse_train)



def LinearR(x,y):
	linreg = LinearRegression()
	linreg.fit(x,y)
	p = linreg.predict(x)
	print("LinearR result:\n")
	# We can view the regression coefficients
	print ('Regression Coefficients: \n', linreg.coef_)
	printError(p,y)
	plot_result(p, y)

def LinearR_SGD(x,y):
	from sklearn.preprocessing import StandardScaler	
	# SGD is very senstitive to varying-sized feature values. So, first we need to do feature scaling.
	scaler = StandardScaler()
	scaler.fit(x)
	x_s = scaler.transform(x)
	sgdreg = SGDRegressor(penalty='l2', alpha=0.25, n_iter=200)
	# Compute RMSE on training data
	sgdreg.fit(x_s,y)
	p = sgdreg.predict(x_s)
	print("SGD result:\n")
	printError(p,y)
	plot_result(p, y)

	kf = KFold(n_splits=2)
	xval_err = 0
	for train, test in kf.split(x):
	    scaler = StandardScaler()
	    scaler.fit(x[train])  # Don't cheat - fit only on training data
	    xtrain_s = scaler.transform(x[train])
	    xtest_s = scaler.transform(x[test])  # apply same transformation to test data
	    sgdreg.fit(xtrain_s,y[train])
	    p = sgdreg.predict(xtest_s)	
	    e = p-y[test]
	    xval_err += np.dot(e,e)	
	rmse_10cv = np.sqrt(xval_err/len(x))		    
	print(rmse_10cv)



if __name__ == "__main__":
	data, target = load_data_from_txt()
	verify_data(data,target)
	LinearR(data,target)
	LinearR_SGD(data,target)















