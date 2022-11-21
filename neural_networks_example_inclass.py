import numpy
import math
import random
import matplotlib.pyplot

# create data from a multivariate normal distribution
mean1 = [-2,0,1]
mean2 = [2,0,1]

cov = [[4,0,0],[0,5,0],[0,0,3]]

x1 = numpy.random.multivariate_normal(mean1, cov, 1000)
matplotlib.pyplot.scatter(x1[:,0], x1[:,1], c = 'b', marker = '.')
x2 = numpy.random.multivariate_normal(mean2, cov, 1000)
matplotlib.pyplot.scatter(x2[:,0], x2[:,1], c = 'r', marker = '.')
matplotlib.pyplot.show()

X = numpy.concatenate((x1,x2))
X = numpy.concatenate((numpy.ones((2000,1)),X), axis = 1)



#first half of the data from class 0, second half from class 1
Xc = numpy.zeros(1000)
Xc = numpy.concatenate((Xc, numpy.ones(1000)))

# randomize the training data

indexes = list(range(2000))
random.shuffle(indexes)

Xshuffled = X[indexes]
Xcshuffled = Xc[indexes]

#Xshuffled = X
#Xcshuffled = Xc


#randomly initialize the weights
#random.seed()
w1 = numpy.transpose(numpy.array([2*random.random()-1, 2*random.random()-1, 2*random.random()-1]))
w2 = numpy.transpose(numpy.array([2*random.random()-1, 2*random.random()-1, 2*random.random()-1]))
w3 = numpy.transpose(numpy.array([2*random.random()-1, 2*random.random()-1, 2*random.random()-1]))

alpha = 0.00001
outputError = numpy.empty((2000,1))
epochError = numpy.empty((500,1))
for epoch in range(500):
    totalError = 0
    for i in range(2000):
        #forward calculation
        z1 = 1/(1 + math.exp(-numpy.dot(Xshuffled[i,:],w1)))
        z2 = 1/(1 + math.exp(-numpy.dot(Xshuffled[i,:],w2)))
        xhidden = [1, z1, z2]
        z3 = 1/(1 + math.exp(-numpy.dot(xhidden,w3)))

        # prediction
        prediction = round(z3,0) #one option...
        outputError[i] = abs(Xcshuffled[i] - prediction)
        #comparison = Xc[i] != prediction
        totalError = totalError + numpy.asscalar(outputError[i])

        #backward propagation
        #update the error
        error3 = z3*(1-z3)*(Xcshuffled[i] - z3) #output node 
        error1 = z1*(1-z1)*(error3*w3[1]) #hidden layer node
        error2 = z2*(1-z2)*(error3*w3[2]) #hidden layer

        w3 = w3 + [alpha*error3, alpha*error3*z1, alpha*error3*z2 ]
        w1 = w1 + [alpha*error1, alpha*error1*Xshuffled[i,1], alpha*error1*Xshuffled[i,2]]
        w2 = w2 + [alpha*error2, alpha*error2*Xshuffled[i,1], alpha*error2*Xshuffled[i,2]]

    # for every epoch
    #print("Iteration... ", epoch + 1, "Error = ", totalError, "            ", end = '\r', flush = True)
    epochError[epoch] = totalError
    print("Iteration... ", epoch + 1, "Error = ", totalError)

    
matplotlib.pyplot.plot(epochError)
matplotlib.pyplot.show()

#epochPlot = numpy.arange(100)
#matplotlib.pyplot.plot(epochError[epoch])
#matplotlib.pyplot.show()

x1test = numpy.random.multivariate_normal(mean1, cov, 1000)
x2test = numpy.random.multivariate_normal(mean2, cov, 1000)

Xtest = numpy.concatenate((x1test,x2test))
Xtest = numpy.concatenate((numpy.ones((2000,1)),Xtest), axis = 1)

#first half of the data from class 0, second half from class 1
Xctest = numpy.zeros(1000)
Xctest = numpy.concatenate((Xctest, numpy.ones(1000)))

prediction = numpy.empty(2000)
for i in range(2000):
    #forward calculation
    z1 = 1/(1 + math.exp(-numpy.dot(Xtest[i,:],w1)))
    z2 = 1/(1 + math.exp(-numpy.dot(Xtest[i,:],w2)))
    xhidden = [1, z1, z2]
    z3 = 1/(1 + math.exp(-numpy.dot(xhidden,w3)))
    prediction[i] = round(z3,0)

totalError = numpy.sum(prediction != Xctest)

print("done")


