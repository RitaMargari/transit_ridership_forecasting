#network analysis API

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from visdom import Visdom
#V 04.03.23

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = 1e9

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss- self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            # print(self.min_validation_loss- self.min_delta,validation_loss)
        else:
            # print(self.min_validation_loss- self.min_delta,validation_loss)
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class NNmodel(nn.Module): #NN model class template
   
    def fit(self,city,A,X,OD,between_fts,
    		Avalid=None,Xvalid=None,ODvalid=None,between_fts_valid=None,
    		Atest=None,Xtest=None,ODtest=None,between_fts_test=None, 
            n_epochs = 1000, lr = 0.005, interim_output_freq = 200,
            full_loss_freq = 20, SEED = 1,early_stop=True): #torch fit
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        start_time = time.time()
        self.fitstate = {'epoch' : -1}
        early_stopper = EarlyStopper(patience=1, min_delta=1e-3)
        # early_stopper= EarlyStopper(patience=5, min_delta=0)
        # params = [*self.parameters()]
        # print()
                            # [self.GNNLayers[i].parameters() for i in range(self.GNNLayerNum)]+[self.OutVNN.parameters()]
        optimizer = optim.Adam(self.parameters(), lr=lr[0])
        best_loss = np.Inf
        # self.fitInitHandle()

        t_1run = time.time()

        self.fitstate = {'epoch': 0, 'n_epochs': n_epochs, 'loss': None, 'full_loss': None, 'start_time': start_time, 'Y': None,'loss_list':[]}

        viz = Visdom()
        
        

        viz.line([0.], [0.], win=city,name='train')
        if early_stop:
        	# pass
            viz.line([[0.0]], [0.], win=city,name='valid', opts=dict(title=city, legend=['valid']))
            viz.line([[0.0]], [0.], win=city,name='test', opts=dict(title=city, legend=['test']))
        for epoch in range(n_epochs):
            self.fitstate['epoch'] = epoch
            

            # if self.iteratelearning:
            # optimizer = self.initoptimizer(lr)
            
            optimizer.zero_grad()
            # mh for trip distribution without parital OD
            # Y = self.forward(OD=OD, X = X_b)
            Y = self.forward(A,X,OD,between_fts)
            self.fitstate['Y'] = Y
            # if self.model == 'p':
            #     loss = self.loss(Y,p)

            # else:
            loss = self.loss(Y, OD)
            # print(loss)
            loss.backward()
            optimizer.step()
            self.fitstate['loss'] = loss.item()
            self.fitstate['loss_list'] += [loss.item()]
            viz.line([loss.item()], [epoch], win=city,name='train', update='append')
            # print(loss)
            # print('--------------------------------')
            if Avalid is None:
                if early_stopper_train.early_stop(loss.item()):
                    print('early stop based on training data at '+str(epoch)+' epochs')             
                    break

            else:
                if early_stop:
                    validation_loss = self.evaluate(Avalid,Xvalid,ODvalid,between_fts_valid)
                    test_loss = self.evaluate(Atest,Xtest,ODtest,between_fts_test)
                    # print('validation_loss',validation_loss)
                    viz.line([validation_loss], [epoch], win=city,name='valid', update='append')
                    viz.line([test_loss], [epoch], win=city,name='test', update='append')

                    if early_stopper.early_stop(validation_loss):
                        print('early stop at '+str(epoch)+' epochs')             
                        break

        # self.fitstate['final_loss'] = best_loss.item()
        # self.fitFinalHandle()
        # return self.fitstate['final_loss']
    def evaluate(self, A,X,OD,between_fts): #evaluate model performance on the new data
        OD = torch.FloatTensor(OD)
        OD = OD.to(torch.device("cuda"))
        win = OD.sum(axis = 0,keepdims=True)
        win = win.to(torch.device("cuda"))
        p = OD / win
        with torch.no_grad():
            # print(X)
            Y = self.forward(A,X,OD,between_fts)

            loss = self.loss(Y, OD)
            return loss.item()


