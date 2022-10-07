import csv
# import sklearn
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import pickle

train = False
datas = []
with open("data1.csv", 'r') as file:
    reader = csv.reader(file)
    for data in reader:
        # print(data)
        datas.append(data)
datas = np.asarray(datas)
X = datas[:-3, :-1]
y = datas[:-3, -1]
print(X.shape)
print(y.shape)

if train:
    svm = SVC()
    svm.fit(X, y)
else:
    with open('model.pkl', 'rb') as f:
        svm = pickle.load(f)




# X_test = datas[-3:, :-1]
# y_test = datas[-3:, -1]
# y_pred = svm.predict(X_test)

plot_confusion_matrix(svm, X, y)
plt.show()


if train:
    # save
    with open('models/model.pkl','wb') as f:
        pickle.dump(svm, f)