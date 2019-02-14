from utils import cov
import torch, random, math
from torch.distributions.multivariate_normal import MultivariateNormal
from multiprocessing import Queue
from torch.multiprocessing import Process


class GaussianMMSampler(object):
    '''Gaussian Mixture Model Sampler

    Gaussian mixture models, is a set of n Gaussians,
    each with their mean - ui and covaraiance matrix - 
    sigmai and a set of weights for each Gaussian denoting 
    the probability that if a point were to be sampled from this 
    model, then that percentage would come from Gaussian
    Gi.

    Args:
        weights: The weights of each Gaussian, in the form
            of a list of Tensors
        means: Means/Centers of the Gaussians
        cov_matrices: The covariance matrices of the 
            given Gaussians
    '''
    def __init__(self, weights , mean, cov_matrices):
        self.weights = weights
        self.means = mean
        self.cov_matrices = cov_matrices
        self.no_of_gaussians = len(weights)


    def mixture_sampling(self, no_of_points):
        '''Sampling from the given mixture model

        Given our mixture model, we sample points 
        with weights as probabilities.

        Args:
            no_of_points: The number of points we 
                want to sample

        Returns:
            1. List of tensor points from the mixture
            model
            2. Gaussian IDs for the points
        '''
        points = list()
        gaussian_id = list()

        #iterates through the list of Gaussians to
        # sample points in according to their percentages
        # from each gaussian

        for i in xrange(self.no_of_gaussians):
            # Define Gaussian Sampler
            gs = GaussianSampler(self.means[i], self.cov_matrices[i])
            # Sample points from Gaussian Sampler
            sample_points = gs.sample_list(int(math.ceil(self.weights[i]*no_of_points)))
            points.extend(sample_points)
            for j in xrange(int(math.ceil(self.weights[i]*no_of_points))):
                # Adding the Gaussian Ids for the points
                gaussian_id.append(i+1)
        assert(len(points) >= no_of_points)
        return points, gaussian_id



class GaussianSampler(object):
    '''Gaussian Sampler

    Function which samples from a Gaussian
    given the mean and the covariance of the 
    Gaussian

    Args:
        mean: Mean of the Gaussian
        cov: covariance matrix of the Gaussian
    '''
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        # Define normal function using Pytorch's multivaraiate function
        self.normal_function = MultivariateNormal(self.mean, self.cov)

    def sample(self):
        '''Sampling function 

        Returns:
            Sampled point
        '''
        return self.normal_function.sample()
        

    def __sample(self, q):
        '''Private function for Sampling

        Has an extra argument to place a 
        multiprocessing Queue in order to extract 
        the returned value
        '''
        q.put(self.normal_function.sample())

    def sample_list(self, no_of_points):
        sample_points = list()
        # Sample points separately in each thread
        for i in xrange(no_of_points):
            q = Queue()
            p = Process(target=self.__sample, args=(q,))
            p.start()
            # retrieves point from Queue
            sample_points.append(q.get())
            p.join()
        return sample_points #being returnd as list of points.


class GaussianRSampler(GaussianSampler):
    '''Gaussian Random Sampler

    Randomly creates a Gaussian of a specific
    dimension, which can then be sampled from

    Args:
        dimensions: The number of dimensions for 
            the Gaussian
    '''
    def __init__(self, dimensions):
        self.dimensions = dimensions
        # Initilaise points from the function
        self.points = self.initilize_points()
        # Initialise mean from the generated points
        self.mean = self.initilize_mean()
        # Initialise Covariance Matrix from the given points
        self.cov = self.initilize_cov_matrix()
        self.zero = torch.zeros(self.mean.shape)

    def initilize_points(self):
        return torch.rand(self.dimensions+100, self.dimensions)

    def initilize_mean(self):
        '''Initialise mean using given points

        Returns
            mean
        '''
        k =  torch.mean(self.points, dim=0).view(self.dimensions, )
        assert(k.shape[0] == self.dimensions)
        return k

    def initilize_cov_matrix(self):
        '''Initialise covariance matrix using the given points

        Returns
            covariance matrix
        '''
        # using utils function to find covariance matrix
        k = cov(self.points)
        assert(k.shape[0] == self.dimensions)
        assert(k.shape[1] == self.dimensions)
        return k
        
    def distance(self, point):
        '''Returns L2 Norm of the point from the center
        which is equivalent to euclidean distance
        '''
        return torch.norm(point-self.mean, p=2)

    def sample_distance(self):
        '''Samples distance of a random point
        sampled from the Gaussian
        '''
        point = self.normal_function.sample()
        return torch.norm(point-self.mean)



# j = GaussianSampler(torch.zeros(3,1), torch.eye(3))
# k = j.sample()
# print k