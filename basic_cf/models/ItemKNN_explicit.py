import numpy as np

class ItemKNN_explicit():
    def __init__(self, train, valid, top_k):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.top_k = top_k

        for i, row in enumerate(self.train):
            self.train[i, np.where(row < 0.5)[0]] = np.nan

        self.user_mean = np.nanmean(self.train, axis=1)
        self.user_mean[np.isnan(self.user_mean)] = 0.0
        self.train = self.train - self.user_mean[:,None]

    def fit(self):
        item_item_sim_matrix = np.zeros((self.num_items, self.num_items))
        
        
        for item_i in range(0, self.num_items):
            for item_j in range(item_i+1, self.num_items):
                
                a = self.train[:,item_i]
                b = self.train[:,item_j]
                co_rated = ~np.logical_or(np.isnan(a), np.isnan(b))
                a = np.compress(co_rated, a)
                b = np.compress(co_rated, b)
                
                if len(a) == 0:
                    continue 

                dot_a_b = np.dot(a, b)
                if dot_a_b == 0:
                    continue
                   
                
                sim = dot_a_b / (np.linalg.norm(a) * np.linalg.norm(b))
                item_item_sim_matrix[item_i,item_j] = sim
                
         
                

        self.item_item_sim_matrix = (item_item_sim_matrix + item_item_sim_matrix.T)

    def predict(self, item_id, user_ids):
        predicted_values = []

       

        for one_missing_user in user_ids:
            # item i를 시청한 item들
            rated_items = np.where(~np.isnan(self.train[one_missing_user,:]))[0]
            unsorted_sim = self.item_item_sim_matrix[item_id, rated_items]

            sorted_items = np.argsort(unsorted_sim)
            sorted_items = sorted_items[::-1]
            
            if(self.top_k > len(sorted_items)):
                top_k = len(sorted_items)
            else:
                top_k = self.top_k 
            sorted_items = sorted_items[0:top_k]
            top_k_items = rated_items[sorted_items]
            
            if(top_k == 0):
                predicted_values.append(0.0)
            else:
                
                items_rate = self.train[one_missing_user, top_k_items]
                items_sim = self.item_item_sim_matrix[top_k_items, item_id]

                items_sim[items_sim < 0.0] = 0.0 
                
                if np.sum(items_sim) == 0.0:
                    predicted_rate = self.user_mean[one_missing_user]
                else:
                    predicted_rate = self.user_mean[one_missing_user] + np.sum(items_rate*items_sim)/np.sum(items_sim)

                predicted_values.append(predicted_rate)
            
        

        return predicted_values


