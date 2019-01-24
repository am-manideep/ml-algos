import csv
import numpy as np
import math
import matplotlib.pyplot as plt

with open('trainData.csv', 'r') as trainData:
    reader = csv.reader(trainData, delimiter=',', quotechar='"')
    Xtranspose = [line[1:] for line in reader]

with open('valData.csv', 'r') as valData:
    reader = csv.reader(valData, delimiter=',', quotechar='"')
    XtransposeVal = [line[1:] for line in reader]

with open('testData.csv', 'r') as testData:
    reader = csv.reader(testData, delimiter=',', quotechar='"')
    XtransposeTest = [line[1:] for line in reader]

with open('trainLabels.csv', 'r') as trainLabels:
    reader = csv.reader(trainLabels, delimiter=',', quotechar='"')
    Y = [line[1:] for line in reader]

with open('valLabels.csv', 'r') as valLabels:
    reader = csv.reader(valLabels, delimiter=',', quotechar='"')
    YVal = [line[1:] for line in reader]

Xtranspose = np.asarray(Xtranspose, dtype=float)
X = np.transpose(Xtranspose)
Y = np.asarray(Y, dtype=float)

XtransposeVal = np.asarray(XtransposeVal, dtype=float)
XVal = np.transpose(XtransposeVal)
YVal = np.asarray(YVal, dtype=float)

XtransposeTest = np.asarray(XtransposeTest, dtype=float)
XTest = np.transpose(XtransposeTest)

lambdas = [0.01, 0.1, 0.7, 0.75, 0.793, 0.8, 0.839, 0.85, 0.9, 1, 10, 100, 1000]
# lambdas = [0.7, 0.75, 0.8, 0.85, 0.9]
# lambdas = [0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83]
# lambdas = [0.791, 0.792, 0.793, 0.794, 0.795]
# lambdas = [0.838, 0.839, 0.84, 0.841, 0.842]
# lambdas = [0.8392, 0.8393, 0.8394]
# lambdas = [0.839]
# lambdas = [0.793]


def ridgeRegression(X, Y, lam):
    X = np.concatenate((X, np.asarray([[1]*X.shape[1]], dtype=float)), axis=0)
    Xt = np.transpose(X)
    I = np.identity(X.shape[0])
    C = np.dot(X, Xt) + lam*I
    d = np.dot(X, Y)
    Cinv = np.linalg.inv(C)
    wbar = np.dot(Cinv, d)
    b = wbar[len(wbar)-1][0]
    w = wbar[:-1]
    obj = np.sum(np.square(np.dot(Xt, wbar) - Y)) + lam * np.sum(np.square(w))
    cvErrs = [0]*X.shape[1]
    cvErrs = np.asarray(cvErrs, dtype=float)
    for i in range(0, X.shape[1]):
        xit = [X[:, i]]
        xi = np.transpose(xit)
        yi = Y[i]
        cvErrs[i] = (np.dot(np.transpose(wbar), xi) - yi)/(1 - np.dot(np.dot(xit, Cinv), xi))
    cvErrs = np.transpose([cvErrs])
    return [w, b, obj, cvErrs]


trainErrs = []
valErrs = []
loocvErrs = []
for lam in lambdas:
    [w, b, obj, cvErrs] = ridgeRegression(X, Y, lam)
    wbar = np.concatenate((w, [[b]]), axis=0)
    Xbar = np.concatenate((X, np.asarray([[1]*X.shape[1]], dtype=float)), axis=0)
    Xbart = np.transpose(Xbar)
    XValbar = np.concatenate((XVal, np.asarray([[1]*X.shape[1]], dtype=float)), axis=0)
    XValbart = np.transpose(XValbar)
    trainErrSquared = np.sum(np.square((np.dot(Xbart, wbar) - Y)))
    trainErr = math.sqrt(np.sum(np.square((np.dot(Xbart, wbar) - Y)))/Y.shape[0])
    valErr = math.sqrt(np.sum(np.square((np.dot(XValbart, wbar) - YVal)))/YVal.shape[0])
    loocvErr = math.sqrt(np.sum(np.square(cvErrs))/cvErrs.shape[0])
    trainErrs.append(trainErr)
    valErrs.append(valErr)
    loocvErrs.append(loocvErr)
    regularizationTerm = lam * (np.sum(np.square(w)))
    # wt = np.transpose(w)
    # wtabs = np.absolute(wt)
    # wtabs = wtabs[0]
    # wt = wt[0]
    # for i in range(len(wt)):
    #     print(i, wt[i], wtabs[i])
    # print(sorted(range(len(wtabs)), key=lambda i: wtabs[i])[-10:])
    # print(sorted(range(len(wtabs)), key=lambda i: wtabs[i])[:10])
    # Xtotal = np.concatenate((X, XVal), axis=1)
    # Xtotalbar = np.concatenate((Xtotal, np.asarray([[1]*Xtotal.shape[1]], dtype=float)), axis=0)
    # Xtotalbart = np.transpose(Xtotalbar)
    # Ytotal = np.concatenate((Y, YVal), axis=0)
    # [wtotal, btotal, objtotal, cvErrstotal] = ridgeRegression(Xtotal, Ytotal, lam)
    # Xtotalerr = math.sqrt(np.sum(np.square(cvErrstotal))/cvErrstotal.shape[0])
    # print(lam, trainErr, valErr, loocvErr, Xtotalerr)
    # print(lam, trainErrSquared, trainErr, valErr, loocvErr, regularizationTerm, obj)

# print(trainErrs)
# print(valErrs)
# print(loocvErrs)

plt.plot(lambdas, trainErrs, label="train errors")
plt.plot(lambdas, valErrs, label="validation errors")
plt.plot(lambdas, loocvErrs, label="loocv errors")
plt.xlabel("lambda")
plt.legend()
plt.show()


def predictY(X, Y, XTest, lam):
    [w, b, obj, cvErrs] = ridgeRegression(X, Y, lam)
    wBar = np.concatenate((w, [[b]]), axis=0)
    XTestBar = np.concatenate((XTest, np.asarray([[1]*XTest.shape[1]], dtype=float)), axis=0)
    YPred = np.dot(np.transpose(wBar), XTestBar)
    return YPred


# bestLam = 0.839
# bestLam = 0.793
# Xtotal = np.concatenate((X, XVal), axis=1)
# Ytotal = np.concatenate((Y, YVal), axis=0)
# Y = predictY(X, Y, XTest, bestLam)
# Y = Y[0]
# i=0
# print('Id,Prediction')
# for y in Y:
#     print(str(i)+','+str(y))
#     i = i+1
