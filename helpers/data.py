"""
    Useful I/O Functions

    Author: (James K J) kj.james2010@vitalum.ac.in

"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

def load_data(filename,return_labels=True):
    """ Load data from csv files and return them in numpy format """
    logging.info("loading data from %s",filename)
    data = np.loadtxt(open("data/"+filename),delimiter=',',
            usecols=range(1,10),skiprows=1,dtype=int)

    if return_labels:
        labels = np.loadtxt(open("data/" +filename),delimiter=',',
                usecols=[0],skiprows=1)
        return labels, data

    return data





