#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
torch.set_printoptions(sci_mode=False)
import networkx as nx

# from GNNAPI import EarlyStopper
from torch_geometric.nn import GATConv, GCNConv
#from NetRepLearnerV2 import NetGNNAttRepr
from NetRepLearner_mh import NetReprLearning
# from NetRepLearner_mh import GMLearning
import pandas as pd
from netAPI import pickleLoad, pickleDump

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist,pdist
import scipy.stats
# from GNNAPI import DirectTensor
import scipy.optimize
import scipy.stats as stats


import time

import geopandas as gpd
from shapely.geometry import Point

import libpysal

from sklearn.preprocessing import StandardScaler

import torch.nn as nn
import matplotlib.pyplot as plt
import scipy

import torch.nn.functional as F
import torch.optim as optim
from visdom import Visdom
import warnings
from csv import writer
warnings.filterwarnings('ignore')


# In[2]:


def inbetween(o,d,A,win,wout):
    distance = A[o,d]
    
    if win[A[o,:]<distance].shape[0]>0: 
        # number of available jobs in a shorter distance
        ofts = win[A[o,:]<distance].sum(axis=0,keepdims=True)[0]
    else:
        ofts = 0
    if wout[A[d,:]<distance].shape[0]>0: 
        # number of available residences in a shorter distance
        dfts = wout[A[d,:]<distance].sum(axis=0,keepdims=True)[0]
    else:
        dfts = 0
#     print(ofts,dfts,win[d],wout[o])
    ofts = ofts/(win[d]+1)
    dfts = dfts/(wout[o]+1)
    between_fts = torch.FloatTensor([ofts,dfts,distance]).view(1,3)
#     print(between_fts)
    return between_fts


# In[3]:



class EarlyStopper2:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.min_gap = 1e9
        self.updated_time = 0

    def early_stop(self, source_loss,target_loss,source_ad_loss,target_ad_loss):
        if np.abs(source_loss-source_ad_loss) + np.abs(source_loss-source_ad_loss)<self.min_gap:
            self.min_gap = np.abs(source_loss-source_ad_loss) + np.abs(source_loss-source_ad_loss)
            self.counter = 0
            self.updated_time += 1
        else:
            self.counter += 1
            if self.counter >= self.patience and self.updated_time>1:
                return True
        return False


# In[4]:


class AdversarialTrainer:
    def __init__(self, input_dim, feature_dim, learning_rate=1e-3):
        torch.manual_seed(0)
        # Initialize models
        self.feature_extractor = self.FeatureExtractor(input_dim, feature_dim)
        self.domain_discriminator = self.DomainDiscriminator(input_dim,feature_dim)

        # Loss and Optimizers
        self.criterion = nn.BCELoss()
        self.optimizer_F = optim.Adam(self.feature_extractor.parameters(), lr=learning_rate)
        self.optimizer_D = optim.Adam(self.domain_discriminator.parameters(), lr=learning_rate)

    class FeatureExtractor(nn.Module):
        def __init__(self, input_dim, feature_dim):
            super().__init__()
            self.model1 = nn.Sequential(
                nn.Linear(input_dim, feature_dim),
                nn.ReLU(),

            )
            self.model2 = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU())


        def forward(self, x):
            
            e = self.model1(x)
            e = self.model2(e)+e
#             e = self.model3(e)
            return e

    class DomainDiscriminator(nn.Module):
        def __init__(self,input_dim, feature_dim):
            super().__init__()
            self.model1 = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, 1),
                nn.Sigmoid()
            )
            self.model2 = nn.Sequential(
                nn.Linear(feature_dim, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.model1(x) + x
            x = self.model2(x)
            return x
    def cuda(self):
        self.feature_extractor = self.feature_extractor.cuda()
        self.domain_discriminator = self.domain_discriminator.cuda()

    def train(self, source_data, target_data, source_labels, target_labels, num_epochs=2000):
        early_stopper = EarlyStopper2(patience=5)
#         viz = Visdom()
#         viz.line([0.], [0.], win='fts extractor',name='source extractor',
#                  opts=dict(legend=['source extractor']))
#         viz.line([0.], [0.], win='fts extractor',name='target extractor',
#                 opts=dict(legend=['target extractor']))
#         viz.line([0.], [0.], win='fts extractor',name='source discriminator',
#                 opts=dict(legend=['source discriminator']))
#         viz.line([0.], [0.], win='fts extractor',name='target discriminator',
#                 opts=dict(legend=['target discriminator']))
        for epoch in range(num_epochs):
            # Train feature extractor and domain discriminator
            self.optimizer_F.zero_grad()
            self.optimizer_D.zero_grad()

            # Source domain
            source_features = self.feature_extractor(source_data)
            source_domain_preds = self.domain_discriminator(source_features)
            loss_source = self.criterion(source_domain_preds, (1-source_labels))
#             viz.line([loss_source.item()], [epoch], win='fts extractor',name='source extractor', update='append')

            loss_source.backward()
            self.optimizer_F.step()
            self.optimizer_F.zero_grad()
            # Target domain
            target_features = self.feature_extractor(target_data)
            target_domain_preds = self.domain_discriminator(target_features)
            loss_target = self.criterion(target_domain_preds, (1-target_labels))
#             viz.line([loss_target.item()], [epoch], win='fts extractor',name='target extractor', update='append')

            loss_target.backward()
            self.optimizer_F.step()
            self.optimizer_F.zero_grad()
#             loss = loss_source + loss_target
#             loss.backward()
            

            source_features = self.feature_extractor(source_data)
            source_domain_preds = self.domain_discriminator(source_features)
            loss_source_adversarial = self.criterion(source_domain_preds, source_labels)
            loss_source_adversarial.backward()
            self.optimizer_D.step()
            self.optimizer_D.zero_grad()
#             viz.line([loss_source_adversarial.item()], [epoch], win='fts extractor',name='source discriminator', update='append')

            target_features = self.feature_extractor(target_data)
            target_domain_preds = self.domain_discriminator(target_features)
            loss_target_adversarial = self.criterion(target_domain_preds, target_labels)
            loss_target_adversarial.backward()
            self.optimizer_D.step()
            self.optimizer_D.zero_grad()
#             viz.line([loss_target_adversarial.item()], [epoch], win='fts extractor',name='target discriminator', update='append')

#             loss_adversarial = loss_source_adversarial + loss_target_adversarial
#             loss_adversarial.backward()
#             self.optimizer_D.step()
#             viz.line([loss_adversarial.item()], [epoch], win='fts extractor',name='discriminator', update='append')

#             viz.line([validation_loss], [epoch], win='fts extractor',name='validation_loss', update='append')

            if early_stopper.early_stop(loss_source.item(),loss_target.item(),
                                        loss_source_adversarial.item(),loss_target_adversarial.item()):
                print('feature extractor early stop at '+str(epoch)+' epochs')             
                break


# In[5]:


class GMLearning(NetReprLearning): #mobility model learning class

    def initNNs(self,fts, GNNConfigs, VNNConfig):
        torch.manual_seed(0)
        self.embed_dim = GNNConfigs['out_features']
        # mh edited, previous infeautre for GNN layer was N nodes as X is none
        
        # mh: v['out_features'] for v in GNNConfigs: self.N, ed
        featuredims = [fts.shape[1]] + [GNNConfigs['out_features']]
        print('featuredims',featuredims)
        self.embed_dim = featuredims[-1]
        self.GNNLayerNum = len(GNNConfigs)
#         self.GNNLayers = [GNN_VNN_Layer(in_features = featuredims[i], 
#                         VNNConfig = GNNConfigs[i]) for i in range(self.GNNLayerNum)]
        self.num_heads = GNNConfigs['transformer_num_heads']
        self.gat_num_heads = GNNConfigs['gat_num_heads']
        self.input_dim = featuredims[0]
        self.attention_out_dim = GNNConfigs['attention_out_dim']
        self.edge_dim = GNNConfigs['edge_dim']
        

        self.shortcut1 = nn.Linear(self.input_dim, GNNConfigs['layer_dims'][0]* self.gat_num_heads)
        self.shortcut2 = nn.Linear(GNNConfigs['layer_dims'][0]* self.gat_num_heads,
                                   GNNConfigs['out_features'])
        
        self.in_conv = GATConv(self.input_dim, GNNConfigs['layer_dims'][0],edge_dim=self.edge_dim,
                               heads=self.gat_num_heads,concat=True)
#         Hidden layers

        self.hidden_layers = torch.nn.ModuleList()
        for i,hidden in enumerate(GNNConfigs['layer_dims']):
            if i + 1 < len(GNNConfigs['layer_dims']):
                self.hidden_layers.append(
                    GATConv(hidden * self.gat_num_heads, GNNConfigs['layer_dims'][i+1], 
                            heads=self.gat_num_heads,
                             concat=True)
                )
        
        # Output layer
        self.out_conv = GATConv(GNNConfigs['layer_dims'][-1] * self.gat_num_heads, 
                                GNNConfigs['out_features'], heads=self.gat_num_heads, 
                                concat=False)

        # Hidden layers
       
       
        self.OutMLP = nn.Linear(GNNConfigs['out_features']*2,1)

        # Linear layers for query, key, and value projections
        self.query_projection = nn.Parameter(torch.randn(self.embed_dim,self.attention_out_dim,device='cuda'))                                                                                             
        self.key_projection = nn.Parameter(torch.randn(self.embed_dim,self.attention_out_dim,device='cuda'))
        self.expo = nn.Parameter(torch.randn(1,1,device='cuda'))
        
        
        

    def forward(self,A, X,OD,between_fts): #feed-forward computation - embedding and the attractivity scores
        wout = OD.sum(axis = 1,keepdims=True)
        win = OD.sum(axis = 0,keepdims=True)
#         OD[OD==0] = 0.1
#         win[win==0] = 0.2
        W = wout * win / win.sum()
        #         print(type(wout),type(win),type(W))

        wout = wout.to(torch.device("cuda"))
        self.win = win.to(torch.device("cuda"))
        W = W.to(torch.device("cuda"))
#         X = X.to(torch.device("cuda"))
        if self.model == 'p':
            rows = []
            cols = []
            for i in range(OD.shape[0]):
                for j in range(OD.shape[0]):
                    if i != j:
                        rows.append(i)
                        cols.append(j)
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_index = edge_index.to(torch.device("cuda"))
#             print(X.device,edge_index.device,between_fts.device)
            E = self.in_conv(X, edge_index,edge_attr=between_fts)

            E += self.shortcut1(X)
            E = nn.Sigmoid()(E)
#             print('----------ReLU-----------------')
#             print(E)


            for layer in self.hidden_layers:
                E = layer(E, edge_index)+E
                E = nn.Sigmoid()(E)

            E = self.out_conv(E, edge_index)+self.shortcut2(E)

            E = nn.Sigmoid()(E)
        if self.attention:
            seq_len, input_dim = E.size()

            query = torch.matmul(E,self.query_projection)
#             print('==============query==================')
#             print(query)
            key =torch.matmul(E,self.key_projection)
#             print('==============key===================')
#             print(key)
            # Reshape the query, key, and value tensors to enable multi-head attention
            query = query.view(seq_len, self.num_heads,self.attention_out_dim // self.num_heads).permute(1, 0, 2)
            key = key.view(seq_len, self.num_heads, self.attention_out_dim  // self.num_heads).permute(1, 0, 2)

            # Compute the dot product attention scores
            scores = torch.matmul(query, key.permute(0, 2, 1))

            scores = scores / torch.sqrt(torch.tensor(input_dim*2 / self.num_heads, dtype=torch.float32))
            attention_weights = torch.mean(scores,axis=0)
            attention_weights = nn.ReLU()(attention_weights)
            y = attention_weights
        elif self.VNNattraction:
            edge_index = list(zip(sorted(list(range(E.shape[0]))*E.shape[0]),
                                      list(range(E.shape[0]))*E.shape[0]))
            edge_index = np.array(edge_index)

            PE_ = torch.concat([E[edge_index[:,0]],E[edge_index[:,1]]],axis=1)
#             print(E)

            batch_size = 10000
            num_batches = PE_.shape[0] // batch_size
            for i in range(num_batches+1):
                # Generate mini-batch indices
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                if end_idx <= PE_.shape[0]:
                    pass
                else:
                    end_idx = PE_.shape[0]+1

                # Extract mini-batch data and labels
                PE_batch = PE_[start_idx:end_idx]
                if i == 0:
                    y = self.OutMLP.forward(PE_batch)
                    y = nn.Sigmoid()(y)
                    y = F.dropout(y,p=0.2)
                else:
                    y_temp = self.OutMLP.forward(PE_batch)
                    y_temp = nn.Sigmoid()(y_temp)
                    y_temp = F.dropout(y_temp,p=0.2)
                    y = torch.concat([y,y_temp])
            y = y.view(E.shape[0], E.shape[0])
            
        if self.model == 'p':
#             A_tensor = torch.FloatTensor(A)
#             A_tensor = A.to(torch.device("cuda"))
#             print(y)
            y = wout * (self.win  * torch.exp(-y*A)) /                        (self.win  * torch.exp(-y*A)).sum(axis=1,keepdims=True)
  


            
        elif self.model == 'g':
            y = wout * (self.win * torch.exp(self.expo*A) /                        (self.win  * torch.exp(self.expo*A)).sum(axis=1,keepdims=True))


    
        return y

    def logMSE(self, Y, Ytrue): #log MSE loss for the unconstrained model
#         print(Y.device,Ytrue.device,self.W.device,mask.device)
        
        loss_ =  ((torch.log(Ytrue + 1) -                                   torch.log(Y + 1)) ** 2).mean()
#         loss_ = (torch.mul(mask , (Ytrue -Y ) ** 2)).sum() / mask.sum()
        return loss_
    def MSE(self, Y, Ytrue): #log MSE loss for the unconstrained model
#         print(Y.device,Ytrue.device,self.W.device,mask.device)
        loss_ =  ((Ytrue - Y ) ** 2).mean()
#         loss_ = (torch.mul(mask , (Ytrue -Y ) ** 2)).sum() / mask.sum()
        return loss_
    def LL(self, p, Ytrue): #log MSE loss for the unconstrained model
        loss_ =  -(Ytrue*torch.log(p)).mean()
        return loss_

    def KL(self, Y, Ytrue): #log MSE loss for the unconstrained model

        loss_ =  (Ytrue*torch.log((Ytrue+1e-8)/(Y+1e-8))).mean()

#         print(loss_)
        return loss_

    def RAE(self, Y, Ytrue): #log MSE loss for the unconstrained model
#         print(Y.device,Ytrue.device,self.W.device,mask.device)
        ybar = Ytrue.mean()
        loss_ =  (torch.absolute(Ytrue - Y )).sum()/(torch.absolute(Ytrue - ybar )).sum()
#         loss_ = (torch.mul(mask , (Ytrue -Y ) ** 2)).sum() / mask.sum()
        return loss_
    def chi(self, Y, Ytrue): #log MSE loss for the unconstrained model

        loss_ =  (((Ytrue - Y )**2)/(Y+1e-8)).mean()
        return loss_

    def loss(self, Y = None, Ytrue = None): #compute loss

        return self.KL(Y,Ytrue)

    def initEmbed(self, x): #initialize node embedding with some initial coordinates
        self.GNNLayers[0].init_params({'X': x})

# mh edited for mask by nodes
def getTrainTestbyNodes(A, train_p = 0.7, seed = 1): #train-test split of the network edges
    np.random.seed(seed)
    number = np.random.uniform(size = A.shape[0])

    valid = number < (1-train_p)**2
    train = np.array([True if i < train_p and i >= (1-train_p)**2 else False for i in number ])
    test = number > train_p
    

    return train,valid, test

def read_area_from_ct(state,nodes):
    ct_map = gpd.read_file('LEHD/'+state+'.zip')
    ct_map['GEOID'] = ct_map['GEOID'].astype(int)
    ct_map = ct_map.set_index('GEOID')
    ct_map = ct_map.loc[nodes]

    return ct_map.to_crs('ESRI:102008').geometry.area.values

def read_location_from_ct(state,nodes):
    ct_map = gpd.read_file('LEHD/'+state+'.zip')
    ct_map['GEOID'] = ct_map['GEOID'].astype(int)
    ct_map = ct_map.set_index('GEOID')
    ct_map = ct_map.loc[sorted(list(nodes))]
    ct_map = ct_map.to_crs('epsg:4326')
    return np.array(list(zip(ct_map.geometry.centroid.x,
                    ct_map.geometry.centroid.y)))


# In[6]:


def MAPE(v, v_hat, axis=None):
        '''
        Mean absolute percentage error.
        :param v: np.ndarray or int, ground truth.
        :param v_: np.ndarray or int, prediction.
        :param axis: axis to do calculation.
        :return: int, MAPE averages on all elements of input.
        '''
#         mask_0 = (v == 0)
#         percentage = (torch.abs(v_hat - v)) / (torch.abs(v)+1)
        error = torch.abs(v_hat - v)
        v[v==0]=1
        mape = ( error/ torch.abs(v)).mean()
#         if torch.any(mask):
#             masked_array = torch.mean(torch.mul(mask,percentage))  # mask the dividing-zero as invalid
#             result = masked_array.mean(axis=axis)
            
        return mape

def common_part_of_commuters(values1, values2, numerator_only=False):
    if type(values1)  == torch.Tensor:
        pass
    else:
        values1 = torch.FloatTensor(values1)
        values2 = torch.FloatTensor(values2)
    if numerator_only:
        tot = 1.0
    else:
        tot = (torch.sum(values1) + torch.sum(values2))
#         print( np.sum(values1))
    if tot > 0:
        return 2.0 * torch.sum(torch.minimum(values1, values2)) / tot
    else:
        return 0.0
def MAE(Y, Ytrue): #log MSE loss for the unconstrained model

#         print(Y.device,Ytrue.device,self.W.device,mask.device)
    loss_ = torch.abs(Ytrue -Y).mean()
    return loss_



# In[7]:


def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)


# In[ ]:





# In[8]:



splitByNodes = True
externalities = True
# if modelmode='g', conventional singly constrained gravity model, other parameters will be ignored
modelmode = 'p'
# gravi con, modelmode = 'p', attention = True, directEmebdding = False
# GCN + MLP, modelmode = 'p', attention = False, VNNattraction = Truepath = 'LEHD/'
seed = 0
cities = [
    ('New York City', 'ny', ['New York County, NY', 'Queens County, NY','Kings County, NY','Bronx County, NY','Richmond County, NY']),
    ('Los Angeles', 'ca', ['Los Angeles County, CA']),
    ('Chicago', 'il', ['Cook County, IL']),
    ('Houston', 'tx', ['Harris County, TX']),
    ('Boston', 'ma', ['Suffolk County, MA', 'Middlesex County, MA']),
    ('Phoenix', 'az', ['Maricopa County, AZ']),
    ('Philadelphia', 'pa', ['Philadelphia County, PA']),
    ('San Antonio', 'tx', ['Bexar County, TX']),
    ('San Diego', 'ca', ['San Diego County, CA']),
    ('Dallas', 'tx', ['Dallas County, TX']),
    ('San Jose', 'ca', ['Santa Clara County, CA']),
    ('Austin', 'tx', ['Travis County, TX']),
]
t_cities = [
    ('New York City', 'ny', ['New York County, NY', 'Queens County, NY','Kings County, NY','Bronx County, NY','Richmond County, NY']),
    ('Los Angeles', 'ca', ['Los Angeles County, CA']),
    ('Chicago', 'il', ['Cook County, IL']),
    ('Houston', 'tx', ['Harris County, TX']),
    ('Boston', 'ma', ['Suffolk County, MA', 'Middlesex County, MA']),
    ('Phoenix', 'az', ['Maricopa County, AZ']),
    ('Philadelphia', 'pa', ['Philadelphia County, PA']),
    ('San Antonio', 'tx', ['Bexar County, TX']),
    ('San Diego', 'ca', ['San Diego County, CA']),
    ('Dallas', 'tx', ['Dallas County, TX']),
    ('San Jose', 'ca', ['Santa Clara County, CA']),
    ('Austin', 'tx', ['Travis County, TX']),
]


fts_extractor_dim = 46
edge_extractor_dim = 3
GNNConfig2 ={'gat_num_heads':2,'transformer_num_heads':2,'attention_out_dim':8,
                             'out_features': 16, 'layer_dims':[16]*2, 'initSeed': seed, 
                                     'actfuncFinal': torch.nn.Sigmoid(),'edge_dim':edge_extractor_dim}

# early_stopper_train = EarlyStopper(patience=10, min_delta=1e-4)
VNNConfig = {'layer_dims':[8], 'dropout': 0.33, 
                            'initSeed': seed, 'actfuncFinal': nn.ReLU()} #nnSquare()
splitByNodes = True
externalities = True
# if modelmode='g', conventional singly constrained gravity model, other parameters will be ignored
modelmode = 'p'
# gravi con, modelmode = 'p', attention = True, directEmebdding = False
# GCN + MLP, modelmode = 'p', attention = False, VNNattraction = True
attention = True
VNNattraction = False
directEmebdding = False #learn node embedding directly (MLP) or through GNNs
filename = 'GravAttnLEHD-spatial-transfer-radiation-euclidean-GAT-resnet-GAN.csv'

with open(filename, 'w') as f:

    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f)

    # Pass the list as an argument into
    # the writerow()
    writer_object.writerow(['source-target']+[city for city,state, counties in cities])

    # Close the file object
    f.close()
        
for city,state, counties in cities:
    splitSeed = 0
    seed = 0

    mainepochs = 500

    lr = [1e-3]
        

    #some basic stats
    print('loading datasets')
    A_train = torch.load('training/'+city+'_A_train.pt').to(torch.device("cuda"))
    OD_train = torch.load('training/'+city+'_OD_train.pt')
    between_fts_train = torch.load('training/'+city+'_between_fts_train.pt')
    nodefts_train = torch.load('training/'+city+'_nodefts_train.pt')
    

    
    RAE_list = []
    CPC_list = []
    MAE_list = []
    pearson_list = []
    jenson_list = []
    
    for target_city,target_state, target_counties in t_cities:

        print('loading target LEHD:',target_city)
        A_valid = torch.load('training/'+target_city+'_A_valid.pt').to(torch.device("cuda"))
        OD_valid = torch.load('training/'+target_city+'_OD_valid.pt')
        between_fts_valid = torch.load('training/'+target_city+'_between_fts_valid.pt')
        nodefts_valid = torch.load('training/'+target_city+'_nodefts_valid.pt')
        
    
        target_A_test = torch.load('training/'+target_city+'_A_test.pt').to(torch.device("cuda"))
        target_OD_test = torch.load('training/'+target_city+'_OD_test.pt')
        target_between_fts_test = torch.load('training/'+target_city+'_between_fts_test.pt')        
        target_nodefts_test = torch.load('training/'+target_city+'_nodefts_test.pt')
        # standardization
        
        scaler = StandardScaler().fit(nodefts_train)
        nodefts_train_t = torch.FloatTensor(scaler.transform(nodefts_train)).to(torch.device("cuda"))
        nodefts_valid_t = torch.FloatTensor(scaler.transform(nodefts_valid)).to(torch.device("cuda"))
        target_nodefts_test_t = torch.FloatTensor(scaler.transform(target_nodefts_test)).to(torch.device("cuda"))

        bet_scaler = StandardScaler().fit(between_fts_train)
        between_fts_train_t = torch.FloatTensor(bet_scaler.transform(between_fts_train)).to(torch.device("cuda"))
        between_fts_valid_t = torch.FloatTensor(bet_scaler.transform(between_fts_valid)).to(torch.device("cuda"))
        target_between_fts_test_t = torch.FloatTensor(bet_scaler.transform(target_between_fts_test)).to(torch.device("cuda"))

#         feature extractor
        print('-----------feature extractor working-----------------')

        fts_trainer = AdversarialTrainer(input_dim=nodefts_train.shape[1], 
                          feature_dim=fts_extractor_dim, learning_rate=1e-3)
        fts_trainer.cuda()
        source_data = nodefts_train_t
        target_data = torch.concat([target_nodefts_test_t,nodefts_valid_t])
        source_labels = torch.ones(source_data.shape[0], 1).to(torch.device("cuda"))
        target_labels = torch.zeros(target_data.shape[0], 1).to(torch.device("cuda"))
        fts_trainer.train(source_data, target_data, source_labels, target_labels, num_epochs=1000)
        fts_trainer.feature_extractor.eval()
        nodefts_train_t = fts_trainer.feature_extractor(nodefts_train_t).detach()
        nodefts_valid_t = fts_trainer.feature_extractor(nodefts_valid_t).detach()
        target_nodefts_test_t = fts_trainer.feature_extractor(target_nodefts_test_t).detach()



#         edge_fts_trainer = AdversarialTrainer(input_dim=between_fts_train.shape[1], 
#                                  feature_dim=edge_extractor_dim, learning_rate=1e-3)
#         edge_fts_trainer.cuda()
#         source_data = between_fts_train_t
#         target_data = torch.concat([target_between_fts_test_t,between_fts_valid_t])
#         source_labels = torch.ones(source_data.shape[0], 1).to(torch.device("cuda"))
#         target_labels = torch.zeros(target_data.shape[0], 1).to(torch.device("cuda"))
#         edge_fts_trainer.train(source_data, target_data, source_labels, target_labels,num_epochs=1000)
#         edge_fts_trainer.feature_extractor.eval()
#         between_fts_train_t = edge_fts_trainer.feature_extractor(between_fts_train_t).detach()
#         between_fts_valid_t = edge_fts_trainer.feature_extractor(between_fts_valid_t).detach()
#         target_between_fts_test_t = edge_fts_trainer.feature_extractor(target_between_fts_test_t).detach()
        
        NRL = GMLearning(city,nodefts_train_t,GNNConfig2, 
                         VNNConfig,modelmode,directembedding = directEmebdding,
                         attention=attention,VNNattraction=VNNattraction) 
        NRL = NRL.cuda()

        NRL.fit(city,
                A_train,nodefts_train_t,OD_train,between_fts_train_t,
                A_valid,nodefts_valid_t,OD_valid,between_fts_valid_t,
                target_A_test,target_nodefts_test_t,target_OD_test,target_between_fts_test_t,
                n_epochs = mainepochs,lr = lr, 
        interim_output_freq = mainepochs//10)

        with torch.no_grad(): 

            target_y_test = NRL.forward(target_A_test,target_nodefts_test_t,
                                        target_OD_test,target_between_fts_test_t)
            target_y_test = torch.nan_to_num(target_y_test)
        # test nodes performance, edges between train and test nodes


        rae = round(NRL.loss(target_y_test.cpu(),torch.FloatTensor(target_OD_test)).item(),3)
        print('KL', rae)
        cpc = common_part_of_commuters(torch.FloatTensor(target_OD_test),target_y_test.cpu())
        if type(cpc) == float:
            pass
        else:
            cpc = cpc.item()
        cpc = round(cpc,3)
        print('CPC', cpc)
        mae = round(MAE(torch.FloatTensor(target_OD_test),target_y_test.cpu()).item(),3)
        print('MAE',mae)
        pearson = round(scipy.stats.pearsonr(target_OD_test.flatten(),
                                             target_y_test.cpu().detach().numpy().flatten())[0],3)
        print('pearson', pearson)
        jenson = round(scipy.spatial.distance.jensenshannon(target_OD_test.flatten(),
                                                            target_y_test.cpu().detach().numpy().flatten()),3)
        print('jenson',jenson)

        RAE_list += [rae]
        CPC_list += [cpc]
        MAE_list += [mae]
        pearson_list += [pearson]
        jenson_list += [jenson]

        torch.cuda.empty_cache()

    with open(filename, 'a') as f:


        writer_object = writer(f)


        writer_object.writerow([city+' KL']+RAE_list)
        
        writer_object.writerow([city+' CPC']+CPC_list)
        writer_object.writerow([city+' MAE']+MAE_list)
        writer_object.writerow([city+ 'Pea']+pearson_list)
        writer_object.writerow([city+ 'Jen']+jenson_list)

        f.close()


# In[9]:



splitByNodes = True
externalities = True
# if modelmode='g', conventional singly constrained gravity model, other parameters will be ignored
modelmode = 'p'
# gravi con, modelmode = 'p', attention = True, directEmebdding = False
# GCN + MLP, modelmode = 'p', attention = False, VNNattraction = Truepath = 'LEHD/'
seed = 0
cities = [
    ('New York City', 'ny', ['New York County, NY', 'Queens County, NY','Kings County, NY','Bronx County, NY','Richmond County, NY']),
    ('Los Angeles', 'ca', ['Los Angeles County, CA']),
    ('Chicago', 'il', ['Cook County, IL']),
    ('Houston', 'tx', ['Harris County, TX']),
    ('Boston', 'ma', ['Suffolk County, MA', 'Middlesex County, MA']),
    ('Phoenix', 'az', ['Maricopa County, AZ']),
    ('Philadelphia', 'pa', ['Philadelphia County, PA']),
    ('San Antonio', 'tx', ['Bexar County, TX']),
    ('San Diego', 'ca', ['San Diego County, CA']),
    ('Dallas', 'tx', ['Dallas County, TX']),
    ('San Jose', 'ca', ['Santa Clara County, CA']),
    ('Austin', 'tx', ['Travis County, TX']),
]
t_cities = [
    ('New York City', 'ny', ['New York County, NY', 'Queens County, NY','Kings County, NY','Bronx County, NY','Richmond County, NY']),
    ('Los Angeles', 'ca', ['Los Angeles County, CA']),
    ('Chicago', 'il', ['Cook County, IL']),
    ('Houston', 'tx', ['Harris County, TX']),
    ('Boston', 'ma', ['Suffolk County, MA', 'Middlesex County, MA']),
    ('Phoenix', 'az', ['Maricopa County, AZ']),
    ('Philadelphia', 'pa', ['Philadelphia County, PA']),
    ('San Antonio', 'tx', ['Bexar County, TX']),
    ('San Diego', 'ca', ['San Diego County, CA']),
    ('Dallas', 'tx', ['Dallas County, TX']),
    ('San Jose', 'ca', ['Santa Clara County, CA']),
    ('Austin', 'tx', ['Travis County, TX']),
]


fts_extractor_dim = 46
edge_extractor_dim = 3
GNNConfig2 = buildVNNConfig({'gat_num_heads':2,'transformer_num_heads':2,'attention_out_dim':8,
                             'out_features': 16, 'layer_dims':[16]*2, 'initSeed': seed, 
                                     'actfuncFinal': torch.nn.Sigmoid(),'edge_dim':edge_extractor_dim})

# early_stopper_train = EarlyStopper(patience=10, min_delta=1e-4)
VNNConfig = buildVNNConfig({'layer_dims':[8], 'dropout': 0.33, 
                            'initSeed': seed, 'actfuncFinal': nn.ReLU()}) #nnSquare()
splitByNodes = True
externalities = True
# if modelmode='g', conventional singly constrained gravity model, other parameters will be ignored
modelmode = 'p'
# gravi con, modelmode = 'p', attention = True, directEmebdding = False
# GCN + MLP, modelmode = 'p', attention = False, VNNattraction = True
attention = True
VNNattraction = False
directEmebdding = False #learn node embedding directly (MLP) or through GNNs
filename = 'GravAttnLEHD-spatial-transfer-radiation-euclidean-GAT-resnet.csv'

with open(filename, 'w') as f:

    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f)

    # Pass the list as an argument into
    # the writerow()
    writer_object.writerow(['source-target']+[city for city,state, counties in cities])

    # Close the file object
    f.close()
        
for city,state, counties in cities:
    splitSeed = 0
    seed = 0

    mainepochs = 500

    lr = [1e-3]
        

    #some basic stats
    print('loading datasets')
    A_train = torch.load('training/'+city+'_A_train.pt').to(torch.device("cuda"))
    OD_train = torch.load('training/'+city+'_OD_train.pt')
    between_fts_train = torch.load('training/'+city+'_between_fts_train.pt')
    nodefts_train = torch.load('training/'+city+'_nodefts_train.pt')
    
    A_valid = torch.load('training/'+city+'_A_valid.pt').to(torch.device("cuda"))
    OD_valid = torch.load('training/'+city+'_OD_valid.pt')
    between_fts_valid = torch.load('training/'+city+'_between_fts_valid.pt')
    nodefts_valid = torch.load('training/'+city+'_nodefts_valid.pt')
    
    RAE_list = []
    CPC_list = []
    MAE_list = []
    pearson_list = []
    jenson_list = []
    
    for target_city,target_state, target_counties in t_cities:

        print('loading target LEHD:',target_city)
    
        target_A_test = torch.load('training/'+target_city+'_A_test.pt').to(torch.device("cuda"))
        target_OD_test = torch.load('training/'+target_city+'_OD_test.pt')
        target_between_fts_test = torch.load('training/'+target_city+'_between_fts_test.pt')        
        target_nodefts_test = torch.load('training/'+target_city+'_nodefts_test.pt')
        # standardization
        
        scaler = StandardScaler().fit(nodefts_train)
        nodefts_train_t = torch.FloatTensor(scaler.transform(nodefts_train)).to(torch.device("cuda"))
        nodefts_valid_t = torch.FloatTensor(scaler.transform(nodefts_valid)).to(torch.device("cuda"))
        target_nodefts_test_t = torch.FloatTensor(scaler.transform(target_nodefts_test)).to(torch.device("cuda"))

        bet_scaler = StandardScaler().fit(between_fts_train)
        between_fts_train_t = torch.FloatTensor(bet_scaler.transform(between_fts_train)).to(torch.device("cuda"))
        between_fts_valid_t = torch.FloatTensor(bet_scaler.transform(between_fts_valid)).to(torch.device("cuda"))
        target_between_fts_test_t = torch.FloatTensor(bet_scaler.transform(target_between_fts_test)).to(torch.device("cuda"))

        
        NRL = GMLearning(city,nodefts_train_t,GNNConfig2, 
                         VNNConfig,modelmode,directembedding = directEmebdding,
                         attention=attention,VNNattraction=VNNattraction) 
        NRL = NRL.cuda()

        NRL.fit(city,
                A_train,nodefts_train_t,OD_train,between_fts_train_t,
                A_valid,nodefts_valid_t,OD_valid,between_fts_valid_t,
                target_A_test,target_nodefts_test_t,target_OD_test,target_between_fts_test_t,
                n_epochs = mainepochs,lr = lr, 
        interim_output_freq = mainepochs//10)

        with torch.no_grad(): 

            target_y_test = NRL.forward(target_A_test,target_nodefts_test_t,
                                        target_OD_test,target_between_fts_test_t)
            target_y_test = torch.nan_to_num(target_y_test)
        # test nodes performance, edges between train and test nodes


        rae = round(NRL.loss(target_y_test.cpu(),torch.FloatTensor(target_OD_test)).item(),3)
        print('KL', rae)
        46
        cpc = common_part_of_commuters(torch.FloatTensor(target_OD_test),target_y_test.cpu())
        if type(cpc) == float:
            pass
        else:
            cpc = cpc.item()
        cpc = round(cpc,3)
        print('CPC', cpc)
        mae = round(MAE(torch.FloatTensor(target_OD_test),target_y_test.cpu()).item(),3)
        print('MAE',mae)
        pearson = round(scipy.stats.pearsonr(target_OD_test.flatten(),
                                             target_y_test.cpu().detach().numpy().flatten())[0],3)
        print('pearson', pearson)
        jenson = round(scipy.spatial.distance.jensenshannon(target_OD_test.flatten(),
                                                            target_y_test.cpu().detach().numpy().flatten()),3)
        print('jenson',jenson)

        RAE_list += [rae]
        CPC_list += [cpc]
        MAE_list += [mae]
        pearson_list += [pearson]
        jenson_list += [jenson]

        torch.cuda.empty_cache()

    with open(filename, 'a') as f:


        writer_object = writer(f)


        writer_object.writerow([city+' KL']+RAE_list)
        
        writer_object.writerow([city+' CPC']+CPC_list)
        writer_object.writerow([city+' MAE']+MAE_list)
        writer_object.writerow([city+ 'Pea']+pearson_list)
        writer_object.writerow([city+ 'Jen']+jenson_list)

        f.close()


# In[8]:



splitByNodes = True
externalities = True
# if modelmode='g', conventional singly constrained gravity model, other parameters will be ignored
modelmode = 'p'
# gravi con, modelmode = 'p', attention = True, directEmebdding = False
# GCN + MLP, modelmode = 'p', attention = False, VNNattraction = Truepath = 'LEHD/'
seed = 0
cities = [
    ('New York City', 'ny', ['New York County, NY', 'Queens County, NY','Kings County, NY','Bronx County, NY','Richmond County, NY']),
    ('Los Angeles', 'ca', ['Los Angeles County, CA']),
    ('Chicago', 'il', ['Cook County, IL']),
    ('Houston', 'tx', ['Harris County, TX']),
    ('Boston', 'ma', ['Suffolk County, MA', 'Middlesex County, MA']),
    ('Phoenix', 'az', ['Maricopa County, AZ']),
    ('Philadelphia', 'pa', ['Philadelphia County, PA']),
    ('San Antonio', 'tx', ['Bexar County, TX']),
    ('San Diego', 'ca', ['San Diego County, CA']),
    ('Dallas', 'tx', ['Dallas County, TX']),
    ('San Jose', 'ca', ['Santa Clara County, CA']),
    ('Austin', 'tx', ['Travis County, TX']),
]
t_cities = [
    ('New York City', 'ny', ['New York County, NY', 'Queens County, NY','Kings County, NY','Bronx County, NY','Richmond County, NY']),
    ('Los Angeles', 'ca', ['Los Angeles County, CA']),
    ('Chicago', 'il', ['Cook County, IL']),
    ('Houston', 'tx', ['Harris County, TX']),
    ('Boston', 'ma', ['Suffolk County, MA', 'Middlesex County, MA']),
    ('Phoenix', 'az', ['Maricopa County, AZ']),
    ('Philadelphia', 'pa', ['Philadelphia County, PA']),
    ('San Antonio', 'tx', ['Bexar County, TX']),
    ('San Diego', 'ca', ['San Diego County, CA']),
    ('Dallas', 'tx', ['Dallas County, TX']),
    ('San Jose', 'ca', ['Santa Clara County, CA']),
    ('Austin', 'tx', ['Travis County, TX']),
]


fts_extractor_dim = 46
edge_extractor_dim = 3
GNNConfig2 = buildVNNConfig({'gat_num_heads':2,'transformer_num_heads':2,'attention_out_dim':8,
                             'out_features': 16, 'layer_dims':[16]*2, 'initSeed': seed, 
                                     'actfuncFinal': torch.nn.Sigmoid(),'edge_dim':edge_extractor_dim})

# early_stopper_train = EarlyStopper(patience=10, min_delta=1e-4)
VNNConfig = buildVNNConfig({'layer_dims':[8], 'dropout': 0.33, 
                            'initSeed': seed, 'actfuncFinal': nn.ReLU()}) #nnSquare()
splitByNodes = True
externalities = True
# if modelmode='g', conventional singly constrained gravity model, other parameters will be ignored
modelmode = 'g'
# gravi con, modelmode = 'p', attention = True, directEmebdding = False
# GCN + MLP, modelmode = 'p', attention = False, VNNattraction = True
attention = False
VNNattraction = False
directEmebdding = False #learn node embedding directly (MLP) or through GNNs
filename = 'gravity-spatial-transfer.csv'

with open(filename, 'w') as f:

    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f)

    # Pass the list as an argument into
    # the writerow()
    writer_object.writerow(['source-target']+[city for city,state, counties in cities])

    # Close the file object
    f.close()
        
for city,state, counties in cities:
    splitSeed = 0
    seed = 0

    mainepochs = 500

    lr = [1e-1]
        

    #some basic stats
    print('loading datasets')
    A_train = torch.load('training/'+city+'_A_train.pt').to(torch.device("cuda"))
    OD_train = torch.load('training/'+city+'_OD_train.pt')
    between_fts_train = torch.load('training/'+city+'_between_fts_train.pt')
    nodefts_train = torch.load('training/'+city+'_nodefts_train.pt')
    
    A_valid = torch.load('training/'+city+'_A_valid.pt').to(torch.device("cuda"))
    OD_valid = torch.load('training/'+city+'_OD_valid.pt')
    between_fts_valid = torch.load('training/'+city+'_between_fts_valid.pt')
    nodefts_valid = torch.load('training/'+city+'_nodefts_valid.pt')
    
    RAE_list = []
    CPC_list = []
    MAE_list = []
    pearson_list = []
    jenson_list = []
    
    for target_city,target_state, target_counties in t_cities:

        print('loading target LEHD:',target_city)
    
        target_A_test = torch.load('training/'+target_city+'_A_test.pt').to(torch.device("cuda"))
        target_OD_test = torch.load('training/'+target_city+'_OD_test.pt')
        target_between_fts_test = torch.load('training/'+target_city+'_between_fts_test.pt')        
        target_nodefts_test = torch.load('training/'+target_city+'_nodefts_test.pt')
        # standardization
        
        scaler = StandardScaler().fit(nodefts_train)
        nodefts_train_t = torch.FloatTensor(scaler.transform(nodefts_train)).to(torch.device("cuda"))
        nodefts_valid_t = torch.FloatTensor(scaler.transform(nodefts_valid)).to(torch.device("cuda"))
        target_nodefts_test_t = torch.FloatTensor(scaler.transform(target_nodefts_test)).to(torch.device("cuda"))

        bet_scaler = StandardScaler().fit(between_fts_train)
        between_fts_train_t = torch.FloatTensor(bet_scaler.transform(between_fts_train)).to(torch.device("cuda"))
        between_fts_valid_t = torch.FloatTensor(bet_scaler.transform(between_fts_valid)).to(torch.device("cuda"))
        target_between_fts_test_t = torch.FloatTensor(bet_scaler.transform(target_between_fts_test)).to(torch.device("cuda"))

        
        NRL = GMLearning(city,nodefts_train_t,GNNConfig2, 
                         VNNConfig,modelmode,directembedding = directEmebdding,
                         attention=attention,VNNattraction=VNNattraction) 
        NRL = NRL.cuda()

        NRL.fit(city,
                A_train,nodefts_train_t,OD_train,between_fts_train_t,
                A_valid,nodefts_valid_t,OD_valid,between_fts_valid_t,
                target_A_test,target_nodefts_test_t,target_OD_test,target_between_fts_test_t,
                n_epochs = mainepochs,lr = lr, 
        interim_output_freq = mainepochs//10)

        with torch.no_grad(): 

            target_y_test = NRL.forward(target_A_test,target_nodefts_test_t,
                                        target_OD_test,target_between_fts_test_t)
            target_y_test = torch.nan_to_num(target_y_test)
        # test nodes performance, edges between train and test nodes


        rae = round(NRL.loss(target_y_test.cpu(),torch.FloatTensor(target_OD_test)).item(),3)
        print('KL', rae)
        cpc = common_part_of_commuters(torch.FloatTensor(target_OD_test),target_y_test.cpu())
        if type(cpc) == float:
            pass
        else:
            cpc = cpc.item()
        cpc = round(cpc,3)
        print('CPC', cpc)
        mae = round(MAE(torch.FloatTensor(target_OD_test),target_y_test.cpu()).item(),3)
        print('MAE',mae)
        pearson = round(scipy.stats.pearsonr(target_OD_test.flatten(),
                                             target_y_test.cpu().detach().numpy().flatten())[0],3)
        print('pearson', pearson)
        jenson = round(scipy.spatial.distance.jensenshannon(target_OD_test.flatten(),
                                                            target_y_test.cpu().detach().numpy().flatten()),3)
        print('jenson',jenson)

        RAE_list += [rae]
        CPC_list += [cpc]
        MAE_list += [mae]
        pearson_list += [pearson]
        jenson_list += [jenson]

        torch.cuda.empty_cache()

    with open(filename, 'a') as f:


        writer_object = writer(f)


        writer_object.writerow([city+' KL']+RAE_list)
        
        writer_object.writerow([city+' CPC']+CPC_list)
        writer_object.writerow([city+' MAE']+MAE_list)
        writer_object.writerow([city+ 'Pea']+pearson_list)
        writer_object.writerow([city+ 'Jen']+jenson_list)

        f.close()


# In[ ]:




