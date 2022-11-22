import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math


#setup constants
C = 0.001

MEAN1 = [-2,0,1]
MEAN2 = [2,0,1]
NUM_DATAPOINTS = 1000
EPOCH = 500
ALPHA = [1,0.5,0.1]
NUM_HIDDEN_LAYER = 3
NUM_OUTPUT_LAYER = 1



#create two classes of data


#cov is 0, variance is 4,5,3
cov = [[4,0,0],[0,5,0],[0,0,3]]

#each should have 1000 data points
x1 = numpy.random.multivariate_normal(MEAN1,cov,NUM_DATAPOINTS)
x2 = numpy.random.multivariate_normal(MEAN2,cov,NUM_DATAPOINTS)


#plotting to see the data for fun
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(x1[:,0],x1[:,1],x1[:,2],c='b', marker='.')
ax.scatter(x2[:,0],x2[:,1],x1[:,2],c='r', marker='.')
plt.show()



#combining the two data clouds
X = numpy.concatenate((x1,x2))
#adding ones as bias
X = numpy.concatenate((numpy.ones((2000,1)),X),axis=1)

#trying to understand how numpy.concatenate works
# df = pandas.DataFrame(X)
# df.to_excel("example/data2.xlsx")

#adding labels
Xc = numpy.zeros(1000)
Xc = numpy.concatenate((Xc, numpy.ones(1000)))

# randomize the training data

indexes = list(range(2000))
random.shuffle(indexes)

Xshuffled = X[indexes]
Xcshuffled = Xc[indexes]

print(Xcshuffled)
#initialize weight
random.seed()

#weights for the hidden layer neurons
#initial trying the range from -1 to 1
w1 = numpy.transpose(numpy.array([2*random.random()-1, 2*random.random()-1, 2*random.random()-1,2*random.random()-1]))
w2 = numpy.transpose(numpy.array([2*random.random()-1, 2*random.random()-1, 2*random.random()-1,2*random.random()-1]))
w3 = numpy.transpose(numpy.array([2*random.random()-1, 2*random.random()-1, 2*random.random()-1,2*random.random()-1]))
#weights for the output layer neuron
w4 = numpy.transpose(numpy.array([2*random.random()-1, 2*random.random()-1, 2*random.random()-1,2*random.random()-1]))


errorList = []

outputError = numpy.empty((2000,1))
epochError = numpy.empty((500,1))
figure = 1
for alpha in ALPHA:
    
    for epoch in range(500):
        totalError = 0
        for i in range(2000):
            #forward calculation
            z1 = 1/(1 + math.exp(-numpy.dot(Xshuffled[i,:],w1)))
            z2 = 1/(1 + math.exp(-numpy.dot(Xshuffled[i,:],w2)))
            z3 = 1/(1 + math.exp(-numpy.dot(Xshuffled[i,:],w3)))
            xhidden = [1, z1, z2,z3]
            z4 = 1/(1 + math.exp(-numpy.dot(xhidden,w4)))

            # prediction
            prediction = round(z4,0) #one option...
            outputError[i] = abs(Xcshuffled[i] - prediction)
            #comparison = Xc[i] != prediction
            # np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
            totalError = totalError + outputError[i].item()

            #backward propagation
            #update the error
            error4 = z4*(1-z4)*(Xcshuffled[i] - z4)#output node
            error1 = z1*(1-z1)*(error4*w4[1]) #hidden layer node
            error2 = z2*(1-z2)*(error4*w4[2]) #hidden layer
            error3 = z3*(1-z3)*(error4*w4[3]) #hidden layer

            w4 = w4 + [alpha*error4, alpha*error4*z1, alpha*error4*z2, alpha*error4*z3]
            w1 = w1 + [alpha*error1, alpha*error1*Xshuffled[i,1], alpha*error1*Xshuffled[i,2], alpha*error1*Xshuffled[i,3]]
            w2 = w2 + [alpha*error2, alpha*error2*Xshuffled[i,1], alpha*error2*Xshuffled[i,2], alpha*error2*Xshuffled[i,3]]
            w3 = w3 + [alpha*error3, alpha*error3*Xshuffled[i,1], alpha*error3*Xshuffled[i,2], alpha*error3*Xshuffled[i,3]]

        # for every epoch
        #print("Iteration... ", epoch + 1, "Error = ", totalError, "            ", end = '\r', flush = True)
        epochError[epoch] = totalError
        print("Iteration... ", epoch + 1, "Error = ", totalError)
    
    plt.figure(figure)
    figure += 1
    plt.plot(epochError)
    


plt.show()

    


#epochPlot = numpy.arange(100)
#matplotlib.pyplot.plot(epochError[epoch])
#matplotlib.pyplot.show()

x1test = numpy.random.multivariate_normal(MEAN1, cov, 1000)
x2test = numpy.random.multivariate_normal(MEAN2, cov, 1000)

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
    z3 = 1/(1 + math.exp(-numpy.dot(Xtest[i,:],w3)))
    xhidden = [1, z1, z2,z3]
    z4 = 1/(1 + math.exp(-numpy.dot(xhidden,w4)))
    prediction[i] = round(z4,0)

totalError = numpy.sum(prediction != Xctest)

print("done")






