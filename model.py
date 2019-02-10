import torch
import random
from torch.distributions.multivariate_normal import MultivariateNormal
from utils import cov

class EstimationMaximisation(object):
    '''Gaussian Estimation Maximisation Algorithm

    The following models the Gaussian Estimation Maximisation algorithm.
    Given a set of points, we do a clustering if them into Gaussian models,
    given that the number of clusters is selected before. 

    Args:
        points: List of Pytorch tensors on which the algorithm
            is to be run
        no_of_iterations: The number of iterations the algorithm
            is to be run on.
        no_of_gaussians: The number of clusters the data is to be
            divided into
        parametric: whether the clustering is soft or hard, 
            where soft clustering is not assigning weights 
            to a point for each cluster it is to belong to, while
            hard is that a point belongs to a single cluster.

    '''
    def __init__(self, points, no_of_iterations, no_of_gaussians, parametric="Yes"):
        '''Initialise variables for EM Model

        Sets the means, weights, gamma and the covaraince 
        matrices to None. 
        '''
        self.points = points #list of pytorch  tensors
        self.no_of_points = len(self.points)
        self.no_of_iterations = no_of_iterations
        self.parametric = parametric
        self.no_of_gaussians = no_of_gaussians
        self.dimension = points[0].shape[0]
        self.means = None
        self.cov_matrices = None
        self.weights = None
        self.gamma = None
        if parametric == 'No':
            raise Warning('Non - Parametric version to be implememted however not available, shifting to Parametric version...')
            self.parametric == 'Yes'

    def initilize_means(self):
        '''Initialises Means for the Gaussians

        Intialises a random Pytorch Tensor of the dimensions
        equal to that of the points dimensions, for
        each Gaussian.

        Returns:
            list of mean Tensors
        '''
        means = list()
        for i in xrange(self.no_of_gaussians):
            means.append(torch.rand(self.dimension, ))
        self.means = means
        return means


    def initilize_cov_matrices(self):
        '''Initialise Covariance Matrices for the Gaussians

        Initialise Covaraiance matrices for the Gaussians
        by first initialising a set of points and then 
        computing the Covaraiance matrix for it.

        Returns:
            List of Tensor Covariance matrices
        '''
        cov_matrices = list()
        for i in xrange(self.no_of_gaussians):
            cov_matrices.append(cov(torch.rand(self.dimension+100, self.dimension)))
        self.cov_matrices = cov_matrices
        assert(cov_matrices[0].shape[0] == self.dimension)
        assert(cov_matrices[0].shape[1] == self.dimension)
        return cov_matrices

    def initialize_parameters(self):
        '''Initialise weights for the Gaussians

        Initialise weights for the Gaussians, given
        the Estimation Maximisation Algorithm
        is "parametric"

        Returns:
            List of Tensor weights
        '''
        params = list()
        for i in xrange(self.no_of_gaussians):
            params.append(torch.rand(1))
        l = sum(params)
        for i in xrange(self.no_of_gaussians):
            params[i] = params[i]/l
        # assert(sum(params) == 1.0)
        self.weights = params
        return params

    def initialize_gamma(self):
        '''Initialise Gamma
        
        Gamma is a matrix of size nxd and can be defined 
        as probabilty of each point belonging to each
        Gaussian

        Returns:
            Gamma Tensor matrix
        '''
        k = torch.rand(self.no_of_points, self.no_of_gaussians)
        sums = torch.sum(k, dim=0)
        for i in xrange(self.no_of_gaussians):
            k[:, i] = k[:, i]/sums[i]
        self.gamma = k
        return k

    def update_parametric(self):
        '''Estimation Maxiisation Update Function

        Firstly step M is done - update the gamma 
        matrix by finding the probabilites of each
        point in each Gaussian. Using this partitiom the 
        centers, weights and the covariance matrices of the Gaussians are 
        updated K-step.
        '''
        gamma = torch.ones((self.no_of_points, self.no_of_gaussians)) 
        # Finding the gamma matrix for the iteration, go through
        # all points for all Gaussians
        for i in xrange(self.no_of_points):
            l = 0.0
            for j in xrange(self.no_of_gaussians):
                # Define Multivariate Function
                normal_function = MultivariateNormal(self.means[j], self.cov_matrices[j])
                # Find the porbability for point
                prob = torch.exp(normal_function.log_prob(self.points[i]))*self.weights[j]
                # Update the gamma fiunction
                gamma[i, j] = prob
                l += prob
            for j in xrange(self.no_of_gaussians):
                # Normalise the Gamma function over a Gaussian
                gamma[i, j] = gamma[i, j]/l
        self.gamma = gamma
        # row wise sum of gamma matrix
        s = torch.sum(gamma, dim=0) 
        for i in xrange(self.no_of_gaussians):
             # updating weights using the weight calculation formula
            self.weights[i] = s[i]/self.no_of_points
            # define a mean tensor
            mean = torch.zeros((self.dimension, ))
            for j in xrange(self.no_of_points): 
                # print np.argwhere(np.isnan(gamma[j, i]))
                # k = self.points[j]*gamma[j, i])

                # Find the weighted mean using points "Parametric EM"
                mean += self.points[j]*gamma[j, i])
            self.means[i] = mean/s[i] #updating means
        for i in xrange(self.no_of_gaussians):
            # update covaraince matrices
            g = torch.tensor(self.points).view(self.no_of_points, self.dimension) - self.means[i].view(1, self.dimension)        
            self.cov_matrices[i] = torch.mm(g.t(), self.gamma[:, i]*g) #updating covariance matrices
        # print self.means


    def update_inverse_parametric(self):
        '''Estimation Maxiisation Update Function inverse

        In order to tackle the low probability of points
        in each Gaussian first the K-step is done, gamma is taken
        at random and first the means and covariance matrices and 
        weights are updated, then gamma is updated

        The first update method is recommended however if not
        then this method should help make the algorthm
        converge.
        '''
        s = torch.sum(self.gamma, dim=0)
        # Initialise weights, means ans=d covriance matrices
        self.weights = list()
        self.means = list()
        self.cov_matrices = list()
        for i in xrange(self.no_of_gaussians):
            self.weights.append(0)
            self.means.append(0)
            self.cov_matrices.append(0)
        # update the means and weights of the Gaussians
        for i in xrange(self.no_of_gaussians):
            self.weights[i] = s[i]/self.no_of_points
            mean = torch.zeros((self.dimension, ))
            for j in xrange(self.no_of_points): 
                # k = np.multiply(self.points[j], self.gamma[j, i])
                mean += self.points[j] * self.gamma[j, i]
            self.means[i] = mean/s[i]
        # update the covariance matrices for the Gaussians
        for i in xrange(self.no_of_gaussians):
            g = torch.tensor(self.points).view(self.no_of_points, self.dimension)-self.means[i].view(1, self.dimension))        
            self.cov_matrices[i] = torch.mm(g.t(), self.gamma[:, i].view(self.gamma.shape[0], 1)*g)
        gamma = torch.ones((self.no_of_points, self.no_of_gaussians)) 
        # Using means, covariance matrices and weights, update gamma
        for i in xrange(self.no_of_points):
            l = 0.0
            for j in xrange(self.no_of_gaussians):
                prob = MultivariateNormal.probs(self.points[i], self.means[j], self.cov_matrices[j], allow_singular=True)*self.weights[j]
                # print 'a'
                gamma[i, j] = prob
                l += prob
            for j in xrange(self.no_of_gaussians):
                gamma[i, j] = gamma[i, j]/l
        self.gamma = gamma
        

    def update_NonParametric(self):
        NotImplemented

    def update_inverse_NonParametric(self):
        NotImplemented


    def iterate(self):
        for i in xrange(self.no_of_iterations):
            if  i == 0:
                if self.means == None:
                    self.initilize_means()
                if self.cov_matrices == None:
                    self.initilize_cov_matrices()
                if self.weights == None:
                    self.initialize_parameters()
            print 'iteration - '+str(i+1)
            if self.parametric == 'Yes':
                self.update_parametric()
            else:
                self.update_NonParametric()
            print ''
            print 'iteration complete'

    def iterate_inverse(self):    
        for i in xrange(self.no_of_iterations):
            if i == 0:
                if self.gamma == None:
                    self.initialize_gamma()
            print 'iteration - '+str(i+1)
            if self.parametric == 'Yes':
                self.update_inverse_parametric()
            else:
                self.update_inverse_NonParametric()
            print ''
        print '#####Iterations complete#######'

