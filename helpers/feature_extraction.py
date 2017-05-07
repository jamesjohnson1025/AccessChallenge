""" feature_extraction.py

Create the requested datasets

Author: James K J (kj.james2010@vitalum.ac.in)

"""

import logging



logger = logging.getLogger(__name__)
subformatter = logging.Formatter(" [%(asctime)s] %(levelname)s\t> %(message)s")



def create_datasets(X,X_test,y,datasets=[],use_cache=True):


    if use_cache:
        logger.info('System enables cache')
        # Check if all files exist. If not, generate the missing ones
        DATASETS = []
        for dataset in datasets:
            try:
                with open("cache/%s.pkl" % dataset,'rb') as f:
                    logger.debug('Reading pickle file:%s' %dataset)
            except IOError:
                logger.warning('Couldn\'t load dataset %s, will generate it',dataset)
                DATASETS.append(dataset.split('_')[0])
    else:
        logger.info('System disabled feature cache')
        DATASETS = ['basic','tuples','triples','greedy',
                    'greedy2','greedy3']


