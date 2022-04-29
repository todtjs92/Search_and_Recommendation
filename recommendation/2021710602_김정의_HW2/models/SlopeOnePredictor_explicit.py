import numpy as np

class SlopeOnePredictor_explicit():
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]

        for i, row in enumerate(self.train):
            self.train[i, np.where(row < 0.5)[0]] = np.nan  # 값이  0.5보다 작은 측정치는 걍 0으로

    def fit(self):
        """
        You can pre-calculate deviation in here or calculate in predict().
        """
       
                
        
        
        
        pass

    def predict(self, user_id, item_ids): # user_id =  2 , item_ids =  [1,3]

        predicted_values = []
        # user i가 시청한 item들
        rated_items = np.where(~np.isnan(self.train[user_id,:]))[0]
        for one_missing_item in item_ids:
            # ========================= EDIT HERE ========================
            # [user_id, one_missing_item] 구하기
            bunmo = 0
            bunza = 0
            for r in rated_items:

                temp = self.train[:,one_missing_item] - self.train[:,r]
                item_counting = sum( ~np.isnan(temp) )
                temp_mean = np.nanmean(temp)

                if np.isnan(temp_mean):
                    continue
                else:

                    user_plus_mean =self.train[user_id,r]+ temp_mean
                    bunza  += user_plus_mean * item_counting
                    bunmo  += item_counting

            predicted_values.append( bunza/bunmo)

                      
            # ============================================================
        return predicted_values


    
    
    
    
    
    
    
    
    
    
    
