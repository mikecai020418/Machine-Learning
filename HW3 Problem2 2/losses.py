### Losses
import numpy as np
class MSE:
    def loss(self, predictions, y):

        l = np.mean((y - predictions)**2)
        return l
    def loss_gradient(self, predictions, y):
        grad = -2*(y - predictions)/y.shape[1]
        return grad.T


# Problem 2 Part 2: You need to implement the Cross-Entropy loss.
class Cross_Entropy:
    def loss(self, predictions, y):
        #compute here the cross entropy loss l, you may want to add 1e-6 to the argument of the logarithm for numercal stability
        epsilon = 1e-6
        l = -np.mean(y * np.log(predictions + epsilon))
        return l
    
    def loss_gradient(self, predictions, y):
        #compute here the cross entropy loss l as computed before, the result, grad, should be of dimension n x m, where m is the number of points in the batch
        epsilon = 1e-6
        grad = -y / (predictions + epsilon) / y.shape[1]
        return grad.T
