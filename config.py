


# config file for model
# Do not call call this class directly,
# instead overwrite the following class

class Config(object):
    '''
    Configuratio class, use this as information
    for args of the model. In order to change it, 
    call a new class and ovveride the following.
    '''

    # number of iterations for the Gaussian Estimation
    # Maximsation model. The following number of iterations
    # can be set to 10
    NUMBER_OF_ITERATIONS = 10

    # paramatric model of the Gaussian Model
    # will generally be set to "yes" - smooth
    # clustering model

    PARAMETRIC = "yes"

    # datasets for the following model
    # are stored in datasets named "set1"
    # and "set2"

    DATASETS = ['set1', 'set2']


    def __init__(self):
        '''
        Setting the attributes for the following 
        class
        '''
        # Dataset 1 is chosen as default
        self.DATASET = DATASETS[0]


