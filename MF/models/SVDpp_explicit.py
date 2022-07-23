import numpy as np
import torch

class SVDpp_explicit_model(torch.nn.Module):
    def __init__(self, num_users , num_items , n_features):
        super().__init__()
        self.user_factors = torch.nn.Embedding(num_users , n_features , sparse = False)
        self.item_factors = torch.nn.Embedding(num_items , n_features , sparse = False)
        self.user_bias = torch.nn.Embedding(num_users, 1, sparse=False)
        self.item_bias = torch.nn.Embedding(num_items, 1, sparse=False)

        torch.nn.init.normal_(self.user_factors.weight, std = 0.01)
        torch.nn.init.normal(self.item_factors.weight , std = 0.01)
        torch.nn.init.normal_(self.user_bias.weight, std=0.01)
        torch.nn.init.normal_(self.item_bias.weight, std=0.01)

    def forward(self,implicit_train_matrix):
        FY_matrix = torch.matmul(implicit_train_matrix ,self.item_factors.weight  ) # user * K
        reconstruction = torch.matmul(self.user_factors.weight , self.item_factors.weight.T) + torch.matmul(FY_matrix,self.item_factors.weight.T)
        reconstruction = reconstruction + self.user_bias.weight  # add user bias
        reconstruction = reconstruction + self.item_bias.weight.T  # add item bias

        return reconstruction

class SVDpp_explicit():
    def __init__(self, train , valid , n_features= 20 , learning_rate = 1e-2 , reg_lambda = 0.1 , num_epochs= 100):
        self.train = train
        self.valid = valid
        self.num_users = train.shape[0]
        self.num_items = train.shape[1]
        self.num_epochs = num_epochs
        self.n_features = n_features

        self.y = np.zeros_like(self.train)
        for i , row in enumerate(self.train):
            self.y[i, np.where(row > 0.5)[0]] = 1.0

        self.model = SVDpp_explicit_model(self.num_users, self.num_items , self.n_features)#.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate , weight_decay = reg_lambda)

    def mse_loss(self,y,target,predict):
        return (y * ( target - predict) **2 ).sum()


    def fit(self):
        ratings = torch.FloatTensor(self.train)
        weights = torch.FloatTensor(self.y)

        implicit_ratings = torch.FloatTensor(self.train).bool().float()

        small_num = float(1e-10)
        implicit_ratings = implicit_ratings * small_num

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()

            prediction = self.model.forward(implicit_ratings)
            loss = self.mse_loss(weights , ratings , prediction)

            loss.backward()

            self.optimizer.step()

        with torch.no_grad():
            self.reconstructed = self.model.forward(implicit_ratings).cpu().numpy()

    def predict(self,user_id, item_ids):
        return self.reconstructed[user_id, item_ids]
