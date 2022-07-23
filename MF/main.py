from time import time
import numpy as np
from utils import load_data
from utils import eval_explicit
import warnings
import random
import torch
import argparse

from models.MF_explicit import MF_explicit
from models.SVDpp_explicit import SVDpp_explicit
from models.BiasedMF_explicit import BiasedMF_explicit

warnings.filterwarnings('ignore')

def seed_everything(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


if __name__ == '__main__':
    seed = 1
    seed_everything(seed)

    parser = argparse.ArgumentParser(description='Select the model')
    parser.add_argument('--model', type=str, help='MF_explicit ,SVDpp_explicit, BiasedMF_explicit')
    parser.add_argument('--n_features', type=int, help="The number of latent vector's dimension")
    args = parser.parse_args()

    dataset= '../naver_movie_dataset_100k.csv'
    print('Model : ',args.model)
    print('n_features :', args.n_features)

    train_data , valid_data , test_data = load_data(dataset , implicit=False)


    '''
    training
    '''
    print("model training")
    time_start = time()
    if args.model == 'MF_explicit':
        mf = MF_explicit(train = np.copy(train_data), valid = valid_data , n_features = 10)
    elif args.model == 'SVDpp_explicit':
        mf = SVDpp_explicit(train=np.copy(train_data),valid = valid_data , n_features = 10 )
    elif args.model == 'BiasedMF_explicit':
        mf = BiasedMF_explicit(train=np.copy(train_data), valid=valid_data, n_features=10)


    mf.fit()

    print("traing time: " , time()- time_start)

    print("model evaluation")
    mf_rmse = eval_explicit(mf, train_data, test_data)
    print("evaluation time: ", time()-time_start)

    print("RMSE on Test Data")
    print("MF: %f"%(mf_rmse))

