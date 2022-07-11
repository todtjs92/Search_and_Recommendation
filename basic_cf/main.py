# 기본 패키지 import
from time import time
import numpy as np
from utils import load_data
from utils import eval_explicit
import warnings
import random
import warnings
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import os
#from models.User_CF import User_CF


def seed_everything(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

if __name__ == '__main__':
	
    cur_dir = os.getcwd()
    sys.path.insert(0, cur_dir + '/models')
    from models import UserCF

    warnings.filterwarnings('ignore')
    seed = 1
    seed_everything(seed)

    """
    dataset loading
    """
    # put your directory's name	
    dataset= '../naver_movie_dataset_100k.csv'
    #dataset = "../../naver_movie_dataset_small.csv" # "movielens_100k.csv"
    train_data, valid_data, test_data = load_data(dataset, implicit=False)

    """
    model training
    """
    print("model training...")
    time_start = time()
    usercf = UserCF.UserCF(train=np.copy(train_data), valid=valid_data, top_k=10)
    usercf.fit()

   
    """
    model evaluation
    """
    
    print("model evaluation")
    userknn_rmse = eval_explicit(usercf, train_data+valid_data, test_data)
   
    print("RMSE on Test Data")
    print("UserCF: %f"%(userknn_rmse))
    # print("ItemKNN: %f"%(itemknn_rmse))
