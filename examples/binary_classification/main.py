"""

https://github.com/pystruct/pystruct/blob/master/examples/plot_binary_svm.py

Score with pystruct subgradient ssvm: 0.857778 (took 11.426986 seconds)
Score with sklearn and libsvm: 0.920000 (took 0.069885 seconds)

"""

from time import time

import numpy
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

# do a binary digit classification
digits = load_digits()
X, y = digits.data, digits.target

# make binary task by doing odd vs even numers
y = y % 2
# code as +1 and -1
y = 2 * y - 1
X /= X.max()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
numpy.save("train_data.npy", X_train)
numpy.save("train_target.npy", y_train)
numpy.save("test_data.npy", X_test)
numpy.save("test_target.npy", y_test)


from pystruct.models import BinaryClf
from pystruct.learners import (NSlackSSVM, OneSlackSSVM,
                               SubgradientSSVM)


pbl = BinaryClf()
subgradient_svm = SubgradientSSVM(pbl, C=10, learning_rate=0.1, max_iter=100,
                                  batch_size=10)

# online subgradient ssvm
# we add a constant 1 feature for the bias (???)
X_train_bias = numpy.hstack([X_train, numpy.ones((X_train.shape[0], 1))])
X_test_bias = numpy.hstack([X_test, numpy.ones((X_test.shape[0], 1))])

start = time()
subgradient_svm.fit(X_train_bias, y_train)
time_subgradient_svm = time() - start
acc_subgradient = subgradient_svm.score(X_test_bias, y_test)

print("Score with pystruct subgradient ssvm: %f (took %f seconds)"
      % (acc_subgradient, time_subgradient_svm))


libsvm = SVC(kernel='linear', C=10)
start = time()
libsvm.fit(X_train, y_train)
time_libsvm = time() - start
acc_libsvm = libsvm.score(X_test, y_test)
print("Score with sklearn and libsvm: %f (took %f seconds)"
      % (acc_libsvm, time_libsvm))
