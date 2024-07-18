import torch
import torch.optim as optim
import numpy as np
import pickle

from torch import nn


class GravityModel(nn.Module):
    def __init__(self, locations, weights, populations, initial_alpha=1.0, fine_tune_locations=False, lossType = 'MSE', dpow = 2, device='cuda'):
        """
        Initializes the Gravity Model with given locations, weights, populations, and model parameters.

        :param locations: Tensor of shape (n, 2) representing the initial xy coordinates of each location.
        :param weights: Tensor of shape (n,) representing the attractiveness weights of each location.
        :param populations: Tensor of shape (n,) representing the population of each location.
        :param initial_alpha: Initial value for the alpha parameter.
        :param learning_rate: Learning rate for the optimizer.
        :param fine_tune_locations: Boolean to decide if xy coordinates should be fine-tuned.
        """
        self.locations = torch.tensor(locations, requires_grad=fine_tune_locations)
        self.weights = torch.tensor(weights)
        self.populations = torch.tensor(populations)
        self.alpha = torch.tensor([initial_alpha], requires_grad=True)
        self.params = [self.alpha]
        self.lossType = lossType
        self.dpow = dpow
        if fine_tune_locations:
            self.params.append(self.locations)

    def calculate_distances(self):
        """Calculate and return the Euclidean distances between all locations."""
        return torch.cdist(self.locations, self.locations)

    def gravity_matrix(self, distances):
        """
        Construct the gravity model matrix based on the provided distances.
         :param distances: Distance matrix of shape (n, n).
        """
        flows = self.weights * torch.exp(-self.alpha * distances ** self.dpow)
        flow_sums = flows.sum(dim=1, keepdim=True)
        normalized_flows = self.populations[:, None] * flows / flow_sums
        return normalized_flows

    def MSE(self, Y, Ytrue, mask = None, log = True): #log MSE loss for the unconstrained model
        if mask is None:
            mask = Y * 0 + 1
        f = lambda x : torch.log(x + 1) if log else x
        loss = torch.mul(mask , (f(Ytrue) - f(Y)) ** 2).sum() / torch.mul(mask , f(Ytrue) ** 2).sum() #mask.sum()
        return loss

    def binomialB1(self, Y, Ytrue, mask = None): #binomial likelihood loss for the B1 model
        if mask is None:
            mask = Y * 0 + 1
        TM = torch.mul(Y, mask).sum(dim = 1).reshape((-1, 1)) #outflow weights
        PM = torch.div(Y , TM) #normalization to get outflow probabilities
        LL = - torch.mul(mask, torch.mul(Ytrue, torch.log(PM + 1e-8))).sum() / torch.mul(mask, Ytrue).sum() #(mask == 0) + (Ytrue == 0)#add constant to avoid zero probabilities
        return LL

    def loss(self, Y = None, Ytrue = None): #compute loss
        return self.MSE(Y = Y, Ytrue = Ytrue, mask = self.edgebatch, log = self.lossType[:3] == 'log') if self.lossType[-3:] == 'MSE' else self.binomialB1(Y = Y, Ytrue = Ytrue, mask = self.edgebatch)

    def evaluate(self, mask = None, baseline = False): #evaluate model performance on a given sample of edges (mask); baseline will replace the embedding-based attraction Y with a null model Y = 1
        if mask is None:
            mask = np.ones(self.A.shape)
        mask = torch.FloatTensor(mask)
        Ytrue = torch.FloatTensor(self.fullA)
        if baseline == 'const':
            Y = self.W
        elif baseline == 'truemax':
            Y = Ytrue
            Y[Y == 0] = 1e-6
        else:
            Y = self.forward(X = None)
        return self.logMSE(Y = Y, Ytrue = Ytrue, mask = mask, log = self.lossType[:3] == 'log').item() if self.lossType[-3:] == 'MSE' else self.binomialB1(Y = Y, Ytrue = Ytrue, mask = mask).item()

    def fit(self, true_flows, steps=1000, print_every=100, learning_rate=0.01, edgebatching = 0):
        """Fit the model by optimizing the alpha parameter and optionally the locations."""
        true_flows = torch.tensor(true_flows)
        self.optimizer = optim.Adam(self.params, lr=learning_rate)
        self.edgebatch = None
        for step in range(steps):
            self.optimizer.zero_grad()
            distances = self.calculate_distances()
            flows = self.gravity_matrix(distances)
            if edgebatching > 0:
                self.edgebatch = torch.tensor(np.random.uniform(size = flows.shape) < edgebatching)
            loss = self.loss(flows, true_flows)
            if step == 0:
                print(f"Initial loss = {loss.item()}, Alpha = {self.alpha.item()}")
            loss.backward()
            self.optimizer.step()

            if ((step + 1) % print_every == 0) or (step >= steps - 1):
                print(f"Step {step + 1}: Loss = {loss.item()}, Alpha = {self.alpha.item()}")

    def predict(self):
        """Use the trained model to predict the gravity matrix."""
        distances = self.calculate_distances()
        return self.gravity_matrix(distances)

class BatchingGravityModel(GravityModel):
    def fit(self, true_flows, steps=1000, print_every=1, batch_size=32, learning_rate=0.01):
        true_flows = torch.tensor(true_flows)
        self.optimizer = optim.Adam(self.params, lr=learning_rate)
        self.batch_size = batch_size
        distances = self.calculate_distances()
        num_locations = distances.size(0)
        edge_indices = torch.cartesian_prod(torch.arange(num_locations), torch.arange(num_locations))

        for step in range(steps):
            self.optimizer.zero_grad()
            total_loss = 0
            # Shuffle the edge indices to randomize batches
            edge_indices = edge_indices[torch.randperm(edge_indices.size(0))]
            bc = 0
            for i in range(0, edge_indices.size(0), self.batch_size):
                batch_indices = edge_indices[i : i + self.batch_size]
                batch_flows = self.gravity_matrix(distances)[batch_indices[:, 0], batch_indices[:, 1]]
                # Placeholder loss function: sum of batch flows (adjust as needed)
                loss = self.loss(batch_flows, true_flows[batch_indices[:, 0], batch_indices[:, 1]])
                loss.backward()
                total_loss += loss.item()
                bc += 1

            self.optimizer.step()

            if ((step + 1) % print_every == 0) or (step >= steps - 1):
                print(f"Step {step}: Avg Loss = {total_loss / bc}")