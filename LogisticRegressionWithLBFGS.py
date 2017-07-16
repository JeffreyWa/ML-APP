
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets
from sklearn.svm import l1_min_c
from os.path import dirname
from os.path import join


def load_labled_point():
    data = np.empty((100, 1000))
    target = np.empty((100,), dtype=np.int)
    module_path = dirname(__file__)
    lineno = 0
    with open(join(module_path, 'data', 'sample_libsvm_data.txt')) as f:
        for line in f:
           fields = line.strip().split(' ')
           data[lineno][0] = int(fields[0])
           for field in fields[1:]:
               ind = field.split(':')
               data[lineno][int(ind[0])+1] = int(ind[1])
           lineno += 1

    np.random.shuffle(data)
    train_X = data[:80,1:]
    train_y = np.reshape(data[:80,:1],(80,))
    test_X = data[80:,1:]
    test_y = data[80:,:1]

    return train_X, train_y, test_X, test_y 

def compute_coefs():
    X, y, tX, ty = load_labled_point()
    X -= np.mean(X,0)

    cs = l1_min_c(X, y, loss='log') * np.logspace(0, 3)

    start = datetime.now()

    #solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’}, default: ‘liblinear’    
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    coefs_ = []
    for c in cs:
        clf.set_params(C=c)
        clf.fit(X, y)
        coefs_.append(clf.coef_.ravel().copy())
    print("This took ", datetime.now() - start)

    pred_y = clf.predict(tX)
    print('pred',' ','real')
    for i in range(pred_y.shape[0]):
        print(pred_y[i],' ',ty[i][0])

    coefs_ = np.array(coefs_)
    plt.plot(np.log10(cs), coefs_)
    ymin, ymax = plt.ylim()
    plt.xlabel('log(C)')
    plt.ylabel('Coefficients')
    plt.title('Logistic Regression Path')
    plt.axis('tight')
    plt.show()


if __name__ == '__main__':
    compute_coefs()
