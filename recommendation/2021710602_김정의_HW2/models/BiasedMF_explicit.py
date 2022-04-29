import numpy as np
import torch


class BiasedMF_explicit_model(torch.nn.Module):
    def __init__(self, num_users, num_items, n_features):
        super().__init__()
        # ========================= EDIT HERE ========================
        
        # padding_index를 세로로 거는법을 몰라서 유저팩터부분을 거꾸로만들고 계산할때 뒤집겠습니다
        self.user_factors = torch.nn.Embedding( n_features+2, num_users,padding_idx= n_features+1) # sparse=False가 디폴트
        self.item_factors = torch.nn.Embedding(n_features+2 , num_items,padding_idx = n_features)
        
        torch.nn.init.normal_(self.user_factors.weight, std=0.01)
        torch.nn.init.normal_(self.item_factors.weight, std=0.01)
        self.user_factors.weight.data[n_features+1,:]=1.0
        self.item_factors.weight.data[n_features,:]=1.0
        
        

        # ========================= EDIT HERE ========================

    def forward(self):
        reconstruction = None
        # ========================= EDIT HERE ========================
        # 위에서 거꾸로 뒤집어나서 곱할때도 반대로
        reconstruction = torch.matmul(self.user_factors.weight.T, self.item_factors.weight)
        
        # ========================= EDIT HERE ========================
        return reconstruction


class BiasedMF_explicit():
    def __init__(self, train, valid, n_features=20, learning_rate = 1e-2, reg_lambda =0.1, num_epochs = 100):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.num_epcohs = num_epochs
        self.n_features = n_features

        self.y = np.zeros_like(self.train)
        for i, row in enumerate(self.train):
            self.y[i, np.where(row > 0.5)[0]] = 1.0

        self.model = BiasedMF_explicit_model(self.num_users, self.num_items, self.n_features)#.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=reg_lambda)


    def mse_loss(self, y, target, predict):
        return (y * (target - predict) ** 2).sum()


    def fit(self):
        ratings = torch.FloatTensor(self.train)#.cuda()
        weights = torch.FloatTensor(self.y)#.cuda()

        # U와 V를 업데이트 함.
        for epoch in range(self.num_epcohs):
            self.optimizer.zero_grad()

            # 예측
            prediction = self.model.forward()
            loss = self.mse_loss(weights, ratings, prediction)

            # Backpropagate
            loss.backward()

            # Update the parameters
            self.optimizer.step()


        with torch.no_grad():
            self.reconstructed = self.model.forward().cpu().numpy()


    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
