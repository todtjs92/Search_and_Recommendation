import numpy as np
from tqdm import tqdm

class UserCF():
    def __init__(self, train, valid, top_k):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.top_k = top_k
        
        # 계산 쉽게하려고 0점 준것들 nan처리 nanmean으로 계산 쉬움, 뒤에서도 연산 쉬워짐. np.
        for i, row in enumerate(self.train):
            self.train[i, np.where(row < 0.5)[0]] = np.nan
        
        self.user_mean = np.nanmean(self.train, axis=1)
        self.user_mean[np.isnan(self.user_mean)] = 0.0 # 다 nan이어서 nan 되있는거 다시 0으로
        self.train = self.train - self.user_mean[:,None] # user_mean의 shape 바꿔주는 부분 , 그리고 user_mean 빼줌. 
    
    def fit(self):
        user_user_sim_matrix = np.zeros((self.num_users, self.num_users))

        for user_i in tqdm(range(0, self.num_users), desc='user_user_sim_matrix (k=%d)' % self.top_k):
            for user_j in range(user_i+1, self.num_users):
                a = self.train[user_i]
                b = self.train[user_j]

                co_rated = ~np.logical_or(np.isnan(a), np.isnan(b)) # 둘다 nan이 아닌것만 가져옴.
                a = np.compress(co_rated, a) # nan 날리고 있는 애들만 가져와서, dot 
                b = np.compress(co_rated, b)

                if len(a) == 0:
                    continue 

                dot_a_b = np.dot(a, b)
                if dot_a_b == 0:
                    continue
            
                user_user_sim_matrix[user_i, user_j] = dot_a_b / (np.linalg.norm(a) * np.linalg.norm(b))

        self.user_user_sim_matrix = (user_user_sim_matrix + user_user_sim_matrix.T)
        

    def predict(self, user_id, item_ids): # user 하나에 대해 복수 item 예측. 
        predicted_values = []

        for one_missing_item in item_ids:

            # item i를 시청한 사용자들
            rated_users = np.where(~np.isnan(self.train[:, one_missing_item]))[0]
            unsorted_sim = self.user_user_sim_matrix[user_id, rated_users]

            # 유사도 정렬
            sorted_users = np.argsort(unsorted_sim)
            sorted_users = sorted_users[::-1]

            # Top K 이웃 구하기
            if(self.top_k > len(sorted_users)):
                top_k = len(sorted_users)
            else:
                top_k = self.top_k 
            sorted_users = sorted_users[0:top_k]
            top_k_users = rated_users[sorted_users]

            # 예측 값 구하기
            if(top_k == 0):
                predicted_values.append(0.0)
            else:
                users_rate = self.train[top_k_users, one_missing_item]
                users_sim = self.user_user_sim_matrix[user_id, top_k_users]
                
                users_sim[users_sim < 0.0] = 0.0 
                
                if np.sum(users_sim) == 0.0:
                    predicted_rate = self.user_mean[user_id]
                    predicted_values.append(predicted_rate)
                else:
                    predicted_rate = self.user_mean[user_id] + np.sum(users_rate*users_sim)/np.sum(users_sim)
                    predicted_values.append(predicted_rate)

        return predicted_values
