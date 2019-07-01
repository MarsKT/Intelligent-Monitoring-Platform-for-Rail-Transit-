import csv
import numpy as np
def readData():
    X = []
    Y = []
    with open('Housing.csv') as f:
        rdr = csv.reader(f)
        next(rdr)
        for line in rdr:
            xline = [1.0]
            for s in line[:-1]:
                xline.append(float(s))
            X.append(xline)
            Y.append(float(line[-1]))
    return (X,Y)
def showall():
    with open('Housing.csv') as f:
        reader = csv.reader(f)
        for x in reader:
            print(x)
    return
X0,Y0 = readData()
print(X0)
print(len(X0))
print(Y0)
#showall()
d = len(X0) - 10
X = np.array(X0[:d])
y = np.transpose(np.array([Y0[:d]]))
#compute
Xt = np.transpose(X)
XtX = np.dot(Xt,X)
Xty = np.dot(Xt,y)
beta = np.linalg.solve(XtX,Xty)

for data,actual in zip(X0[d:],Y0[d:]):
    x = np.array([data])
    prediction = np.dot(x,beta)
    print('prediction = '+ str(prediction[0,0]) + '     actual = '+ str(actual))