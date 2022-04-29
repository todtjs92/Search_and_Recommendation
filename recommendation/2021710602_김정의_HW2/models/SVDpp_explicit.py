import numpy as np
import torch


class SVDpp_explicit_model(torch.nn.Module):
    def __init__(self, num_users, num_items, n_features):
        super().__init__()
        # ========================= EDIT HERE ========================
        self.item_to_nf = torch.nn.Embedding(num_items,n_features)
        self.item_to_nf_bias = torch.zeros(num_items,2,requires_grad=False)
        
        self.nf_to_item = torch.nn.Embedding(n_features+2,num_items,padding_idx=n_features)
        
        torch.nn.init.normal_(self.item_to_nf.weight, std=0.01)
        torch.nn.init.normal_(self.nf_to_item.weight, std=0.01)
        self.nf_to_item.weight.data[n_features,:]=1.0
        
        

        # ========================= EDIT HERE ========================

    def forward(self, implicit_train_matrix):
        reconstruction = None
        # ========================= EDIT HERE ========================
        # item_to_nf 랑 bias랑 붙이기
        item_to_nf_concat = torch.cat((self.item_to_nf.weight,self.item_to_nf_bias),1)
        
        reconstruction = torch.matmul(implicit_train_matrix,item_to_nf_concat )
        reconstruction = torch.matmul(reconstruction ,self.nf_to_item.weight )
        
        # ========================= EDIT HERE ========================
        return reconstruction


class SVDpp_explicit():
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

        self.model = SVDpp_explicit_model(self.num_users, self.num_items, self.n_features)#.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=reg_lambda)


    def mse_loss(self, y, target, predict):
        return (y * (target - predict) ** 2).sum()


    def fit(self):
        ratings = torch.FloatTensor(self.train)#.cuda()
        weights = torch.FloatTensor(self.y)#.cuda()

        implicit_ratings = torch.FloatTensor(self.train).bool().float() 

        # TODO: normalize implicit ratings with the eplison
        # ========================= EDIT HERE ========================
        small_num = float(1e-10)
        implicit_ratings = implicit_ratings * small_num

        # ========================= EDIT HERE ========================

        # U와 V를 업데이트 함.
        for epoch in range(self.num_epcohs):
            self.optimizer.zero_grad()

            # 예측
            prediction = self.model.forward(implicit_ratings)
            loss = self.mse_loss(weights, ratings, prediction)

            # Backpropagate
            loss.backward()

            # Update the parameters
            self.optimizer.step()


        with torch.no_grad():
            self.reconstructed = self.model.forward(implicit_ratings).cpu().numpy()
        self.implicit_ratings = implicit_ratings.cpu().numpy()

    def predict(self, user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
