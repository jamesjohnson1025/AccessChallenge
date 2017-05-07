from __future__ import division

from sklearn import linear_model,ensemble
from helpers.data import load_data
from helpers.feature_extraction import create_datasets

import argparse
import logging




logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",filename="history.log",filemode='a',level=logging.DEBUG,datefmt='%m/%d/%y %H:%M:%s')

formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",datefmt='%m/%d/%y %H:%M:%s')

console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logger = logging.getLogger(__name__)

def main(CONFIG):
    """
        The final model is a combination of several base models, which are then combined using StackedClassifier defined in the helpers.ml module

        The list of models and associated datasets is generated automatically from their identifying strings. The format is as follows:

        A:b_c where

            A is the initials of the algorithm to use
            b is the base dataset
            c is the feature set and the variants to use

            LR: Linear Regression
    """

    logger.info('--- Executing Model features ---')
    SEED = 42
    selected_models = [
            "LR:tuples_sf",
            "LR:greedy_sfl",
            "LR:greedy2_sfl",
            "LR:greedy3_sf",
            "RFC:basic_b",
            "RFC:tuples_f",
            "RFC:tuples_fd",
            "RFC:greedy_f",
            "RFC:greedy2_f",
            "GBC:basic_f",
            "GBC:tuples_f",
            "LR:greedy_sbl",
            "GBC:greedy_c",
            "GBC:tuples_cf"
            ]


    #Collect all models along with the dataset
    models = []
    for item in selected_models:
        model_id,dataset = item.split(':')
        model = {
                    'LR':linear_model.LogisticRegression,
                    'GBC':ensemble.GradientBoostingClassifier,
                    'RFC':ensemble.RandomForestClassifier,
                    'ETC':ensemble.ExtraTreesClassifier
                }[model_id]()
        model.set_params(random_state=SEED)
        models.append((model,dataset))


    datasets = [dataset for model, datasets in models]
    y,X = load_data('train.csv')
    X_test = load_data('test.csv',return_labels=False)

    logger.info("Preparing datasets (use cache=%s), str(CONFIG.use_cache)")
    create_datasets(X,X_test,y,datasets,CONFIG.use_cache)



if __name__ == '__main__':

   parser = argparse.ArgumentParser(description="Parameters for the script")
   parser.add_argument('-d','--diagnostics',action='store_true',
           help='Compute diagnostics.')
   parser.add_argument('-i','--iter',type=int,default=1,
           help='Number of iterations for averaging.')
   parser.add_argument('-f','--outputfile',default='',
           help='Name of the file where predictions are saved')
   parser.add_argument('-g','--grid-search',action='store_true',
           help='Use grid search to find best parameters')
   parser.add_argument('-m','--model-selection',action='store_true',
           default=False,help='Use model selection')
   parser.add_argument('-n','--no-cache',action='store_false',
           default=True,help='Use cache',dest='use_cache')
   parser.add_argument('-s','--stack',action='store_true',
           help='Use stacking')
   parser.add_argument('-v','--verbose',action='store_true',
           help='show computation steps.')
   parser.add_argument('-w','--fwls',action='store_true',
           help='Use metafeatures.')

   parser.set_defaults(argument_default=False)
   config = parser.parse_args()

   config.stack = config.stack or config.fwls

   logger.debug('\n'+ '='*50)
   main(config)



