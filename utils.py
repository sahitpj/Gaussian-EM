import torch

def meta(dataset):
    '''Extracting Gaussian data as provided in the example problem
    
    Given data is of following format:-
    line1 - no_of_gaussians,dimensions
    line2 - weights (delimter-,)
    line3 for k lines - centers of gaussians (delimter-,)

    Args:
        dataset: The dataset containging the following files. The name of 
            the files however must be in 'centers.txt' name, for it to work

    Returns:
        1. The number of gaussians
        2. The diomension of the vectors
        3. list of weights
        4. The centers in a list of d-dimensional vectors 
    '''
    filepath = 'data/'+dataset+'/centers.txt'
    no_of_gaussians = None
    dimensions = None
    weights = list()
    centers = list()
    f = open(filepath, 'r')
    lines = f.readlines()
    for i in xrange(len(lines)):
        if i == 0:
            lines[i] = map(int, lines[i][:-2].split(','))
            no_of_gaussians, dimensions = lines[0]
        elif i == 1:
            lines[i] = map(float, lines[i][:-3].split(','))
            weights = lines[1]
            assert(sum(weights) == 1.0)
        else:
            centers.append(np.array(map(float, lines[i][:-3].split(','))))
    return no_of_gaussians, dimensions, weights, centers #outputs all data in lists or in values



def covariance_data(no_of_gaussians, dataset):
    '''Method for extracting covariance matrix.

    dxd dimension covariance matrices. It extracts files using numpy and 
    converts them to Pytorch Tensors.

    Args:
        no_of_gaussians: No of gaussians, each gaussian to one covmatrix file
        dataset: The file in which the cov matrices are present

    Returns:
        List of all covariance matrices (Pytorch Tensors).
    '''
    cov_matrices = list()
    for i in xrange(no_of_gaussians):
        filepath = 'data/'+dataset+'/cov_'+str(i+1)+'.txt'
        cov = np.loadtxt(filepath, delimiter=',', comments='#')
        cov_matrices.append(torch.from_numpy(cov))
    return cov_matrices #returned as list of pytorch tensors


def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

