import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
#Machine learning algorithm
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import multilayer_perceptron as MLP
from sklearn.utils import shuffle

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df.head()



#select the proportion, here train/cv = 7:3
row , column = train_df.values.shape
train_df = shuffle(train_df)

#train set
train_set_size = int(row*0.8)
X_train = train_df.iloc[:train_set_size, 1:].values
Y_train = train_df.iloc[:train_set_size, 0].values

#cross-valiation set
cv_set_size = int(row*0.2)
X_cv = train_df.iloc[train_set_size:row, 1:].values
Y_cv = train_df.iloc[train_set_size:row, 0].values

#test set
X_test = test_df.values


print(X_train.shape)
print(X_test.shape)

def show(img):
    plt.imshow(img, cmap="gray", interpolation="none")


# Machine Learning algorithm

#feature scaling
X_train = (X_train/255).astype(sp.float64)
X_cv = (X_cv/255).astype(sp.float64)
X_test = (X_test/255).astype(sp.float64)

#tune
alpha_array = 10.0 ** -np.arange(1, 7)
learing_rate_array = sp.exp(-np.arange(0, 6))



accuray = np.eye(6, 6)

def tune(param1, param2):
    for i in np.arange(0, 6):
        for j in np.arange(0, 6):
            clf = MLPClassifier(solver='adam', alpha=alpha_array[i], learning_rate_init=learing_rate_array[j], random_state=1 )
            clf.fit(X_train, Y_train)
            Y_cv_pred= clf.predict(X_cv)
            evaluate_matrix = (Y_cv_pred == Y_cv).astype(int)
            evaluate_matrix = np.matrix(evaluate_matrix)
            accuray[i, j] = float(evaluate_matrix.sum()/evaluate_matrix.size * 100)


tune(alpha_array, learing_rate_array)
best_i, best_j =  np.unravel_index(accuray.argmax(), accuray.shape)
alpha_p = alpha_array[best_i]
print(alpha_p)
learing_rate_p = learing_rate_array[best_j]
print(learing_rate_p)
# Y_test predict
clf = MLPClassifier(solver='adam', alpha=alpha_p, learning_rate_init=learing_rate_p, random_state=1)
clf.fit(X_train, Y_train)
Y_test = clf.predict(X_test)
print(clf.score(X_train, Y_train))

#export the recognizations
imageId = np.arange(Y_test.size+1)
index = np.delete(imageId, 0)
submission = pd.DataFrame({'ImageId': index,
                        'Label': Y_test})
submission.to_csv('digits_recogniztion.csv',index=False)