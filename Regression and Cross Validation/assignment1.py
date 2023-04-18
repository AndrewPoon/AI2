import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def load(file_name):
    with open(file_name) as file:
        return np.array([float(line.rstrip()) for line in file])

x_train = load('hw1xtr.dat')
y_train = load('hw1ytr.dat')

x_test = load('hw1xte.dat')
y_test = load('hw1yte.dat')

#q2A
plt.scatter(x_train, y_train)
plt.title('Training Data')
plt.show()
plt.scatter(x_test,y_test)
plt.title("Test Data")
plt.show()

def linearRegression(x, y):
    x_transpose = np.transpose(x)
    output = np.matmul(np.linalg.inv(np.matmul(x_transpose, x)), np.matmul(x_transpose, y))
    return output

def error(w, x, y):
    w_transpose = np.transpose(w)
    array = []
    for i in range(len(x)):
        array.append(np.square(np.matmul(w_transpose, x[i]) - y[i]))
    return np.sum(array) / len(x)

def showGraph(title, line,x , y, error):
    indexTrain = np.argsort(x)
    plt.title(title)
    plt.plot(x[indexTrain], line[indexTrain])
    plt.scatter(x, y)
    plt.xlabel("Error: " + str(error))
    plt.show()

x2_train = x_train.reshape(x_train.size, 1)
y2_train = y_train.reshape(y_train.size, 1)
x2_test = x_test.reshape(x_test.size, 1)
y2_test = y_test.reshape(y_test.size, 1)

#q2B + q2C
ones_train = np.ones((x_train.size, 1))
x_lin_train = np.hstack((x2_train, ones_train))
weight = linearRegression(x_lin_train, y2_train)
ones_test = np.ones((x_test.size, 1))
x_lin_test = np.hstack((x2_test, ones_test))
train_line = weight[0] * x_train + weight[1]
test_line = weight[0] * x_test + weight[1]
showGraph("Training model for question 2B",train_line,x_train,y_train,error(weight,x_lin_train,y2_train))
showGraph('Testing model for question 2C',test_line,x_test,y_test,error(weight,x_lin_test,y2_test))

square_train = np.square(x_train).reshape(x_train.size, 1)
x_quad_train = np.hstack((square_train, x2_train, ones_train))
wD = linearRegression(x_quad_train, y2_train)
square_test = np.square(x_test).reshape(x_test.size, 1)
x_quad_test = np.hstack((square_test, x2_test, ones_test))
linetrain_quad = wD[0] * np.square(x_train) + wD[1] * x_train + wD[2]
linetest_quad = wD[0] * np.square(x_test) + wD[1] * x_test + wD[2]
showGraph("Training model for quadratic polynomial regression", linetrain_quad, x_train, y_train, error(wD, x_quad_train, y2_train))
showGraph("Test model for quadartic polynomial regression", linetest_quad, x_test, y_test, error(wD, x_quad_test, y2_test))

cube_train = np.power(x_train, 3).reshape(x_train.size, 1)
x_cube_train = np.hstack((cube_train, square_train, x2_train, ones_train))
wE = linearRegression(x_cube_train, y2_train)
cube_test = np.power(x_test, 3).reshape(x_test.size, 1)
x_cube_test = np.hstack((cube_test, square_test, x2_test, ones_test))
linetrain_cube = wE[0] * np.power(x_train, 3) + wE[1] * np.square(x_train) + wE[2] * x_train + wE[3]
linetest_cube = wE[0] * np.power(x_test, 3) + wE[1] * np.square(x_test) + wE[2] * x_test + wE[3]
showGraph('Training model for cubic polynomial regression',linetrain_cube,x_train,y_train,error(wE,x_cube_train,y2_train))
showGraph('Test model for cubic polynomial regression',linetest_cube,x_test,y_test,error(wE,x_cube_test,y2_test))

quart_train = np.power(x_train, 4).reshape(x_train.size, 1)
x_quart_train = np.hstack((quart_train, cube_train, square_train, x2_train, ones_train))
wF = linearRegression(x_quart_train, y2_train)
quart_test = np.power(x_test, 4).reshape(x_test.size, 1)
x_quart_test = np.hstack((quart_test, cube_test, square_test, x2_test, ones_test))
linetrain_quart = wF[0] * np.power(x_train, 4) + wF[1] * np.power(x_train, 3) + wF[2] * np.square(x_train) + wF[3] * x_train + wF[4]
linetest_quart = wF[0] * np.power(x_test, 4) + wF[1] * np.power(x_test, 3) + wF[2] * np.square(x_test) + wF[3] * x_test + wF[4]
showGraph('Training model for quartic polynomial regression',linetrain_quart,x_train,y_train,error(wF,x_quart_train,y2_train))
showGraph('Test model for quartic polynomial regression',linetest_quart,x_test,y_test,error(wF,x_quart_test,y2_test))

def regularizedLinearRegression(x, y, lmda, size):
    x_transpose = np.transpose(x)
    identity_matrix = np.identity(size)
    identity_matrix[0][0] = 0
    output = np.matmul(np.linalg.inv(np.add(np.matmul(x_transpose, x), identity_matrix * lmda)), np.matmul(x_transpose, y))
    return output

rwInputs = [0.01, 0.05,0.1,0.5,100,1000000]
rw1 = regularizedLinearRegression(x_quart_train, y2_train, 0.01, 5)
rw2 = regularizedLinearRegression(x_quart_train, y2_train, 0.1, 5)
rw3 = regularizedLinearRegression(x_quart_train, y2_train, 1, 5)
rw4 = regularizedLinearRegression(x_quart_train, y2_train, 10, 5)
rw5 = regularizedLinearRegression(x_quart_train, y2_train, 100, 5)
rw6 = regularizedLinearRegression(x_quart_train, y2_train, 1000, 5)
rwTrainingError = [error(rw1, x_quart_train, y2_train), error(rw2, x_quart_train, y2_train), error(rw3, x_quart_train, y2_train), error(rw4, x_quart_train, y2_train), error(rw5, x_quart_train, y2_train), error(rw6, x_quart_train, y2_train)]
rwTestError = [error(rw1, x_quart_test, y2_test), error(rw2, x_quart_test, y2_test), error(rw3, x_quart_test, y2_test), error(rw4, x_quart_test, y2_test), error(rw5, x_quart_test, y2_test), error(rw6, x_quart_test, y2_test)]

#print(rwTrainingError)
#print(rwTestError)
plt.title("Training Errors with different λ")
plt.plot(rwInputs, rwTrainingError)
plt.xscale("log")
plt.show()

plt.title("Test Errors with different λ")
plt.plot(rwInputs, rwTestError)
plt.xscale("log")
plt.show()

weightDataSet = np.hstack((rw1, rw2, rw3, rw4, rw5, rw6))
plt.title("weight parameter")
plt.plot(rwInputs, weightDataSet[0], label = "x^4")
plt.plot(rwInputs, weightDataSet[1], label = "x^3")
plt.plot(rwInputs, weightDataSet[2], label = "x^2")
plt.plot(rwInputs, weightDataSet[3], label = "x^1")
plt.plot(rwInputs, weightDataSet[4], label = "x^0")
plt.xscale("log")
plt.legend()
plt.show()
#print(weightDataSet)
def crossValidation(x_Secs, y_Secs, test_X, test_Y, lambdas, size):
    output = []
    train_x = np.vstack((x_Secs[0],x_Secs[1],x_Secs[2],x_Secs[3]))
    train_y = np.vstack((y_Secs[0],y_Secs[1],y_Secs[2],y_Secs[3]))
      
    for i in lambdas:
        rw = regularizedLinearRegression(train_x, train_y, i, size) 
        output.append(error(rw, test_X, test_Y))
    return output

cVsecX = np.split(x_quart_train, 5)
cVsecY = np.split(y2_train, 5)

cVErrors = []
cVErrors.append(crossValidation([cVsecX[1], cVsecX[2], cVsecX[3], cVsecX[4]], [cVsecY[1], cVsecY[2], cVsecY[3], cVsecY[4]], cVsecX[0], cVsecY[0], rwInputs, 5))
cVErrors.append(crossValidation([cVsecX[0], cVsecX[2], cVsecX[3], cVsecX[4]], [cVsecY[0], cVsecY[2], cVsecY[3], cVsecY[4]], cVsecX[1], cVsecY[1], rwInputs, 5))
cVErrors.append(crossValidation([cVsecX[0], cVsecX[1], cVsecX[3], cVsecX[4]], [cVsecY[0], cVsecY[1], cVsecY[3], cVsecY[4]], cVsecX[2], cVsecY[2], rwInputs, 5))
cVErrors.append(crossValidation([cVsecX[0], cVsecX[1], cVsecX[2], cVsecX[4]], [cVsecY[0], cVsecY[1], cVsecY[2], cVsecY[4]], cVsecX[3], cVsecY[3], rwInputs, 5))
cVErrors.append(crossValidation([cVsecX[0], cVsecX[1], cVsecX[2], cVsecX[3]], [cVsecY[0], cVsecY[1], cVsecY[2], cVsecY[3]], cVsecX[4], cVsecY[4], rwInputs, 5))

cVErrors = np.transpose(cVErrors)
cVErrorsAverage = []
for i in cVErrors:
    cVErrorsAverage.append(np.average(i))

plt.title("five-fold cross-validation average error")
plt.plot(rwInputs, cVErrorsAverage)
plt.xscale("log")
plt.show()
#print(cVErrorsAverage)
lineTestFinal = rw1[0] * np.power(x_test, 4) + rw1[1] * np.power(x_test, 3) + rw1[2] * np.square(x_test) + rw1[3] * x_test + rw1[4]
indexTestFinal = np.argsort(x_test)
plt.title("Best fit (λ = 0.01)")
plt.scatter(x_test, y_test)
plt.plot(x_test[indexTestFinal], lineTestFinal[indexTestFinal])
plt.show()