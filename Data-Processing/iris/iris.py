from mle import MLClassifier
# Load libraries
import pandas
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import model_selection  # 模型比较和选择包
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pltList(resultList):
        x = []
        y = []
        z = []
        # x axis, y axis, z axis.
        for i in range(len(resultList)):
            for j in range(3):
                if j == 0:
                    x.append(resultList[i][j])
                if j == 1:
                    y.append(resultList[i][j])
                if j == 2:
                    z.append(resultList[i][j])

        # plot
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x,y,z)

        # ax.set_zlabel()
        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_zlabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_zlabel('X', fontdict={'size': 15, 'color': 'red'})
        plt.show()

df = pandas.read_csv("iris.data")

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    df.iloc[:, 0:4].values, df.iloc[:, 4].values, train_size=0.8
)

mlc = MLClassifier()
mlc.fit(x_train, y_train)
print("test:",mlc.nclasses)

score = mlc.score(x_test, y_test)

pltList(mlc.likelihoodsList)