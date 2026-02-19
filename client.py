# File contains: Flower client with ALDP-Dx privacy integration
# ** functions/classes
# FlowerClient - implemented, untested, unbackedup
#   input: model(nn.Module), trainloader(DataLoader), privacy_engine(ALDPPrivacyEngine), device(str) | output: FlowerClient object
#   calls: train, get_parameters, set_parameters, fit, evaluate | called by: main.py client_fn
#   process: Flower client that trains locally, sanitizes updates with ALDP-Dx, sends to server

# FlowerClient.get_parameters - implemented, untested, unbackedup
#   input: config(dict) | output: list of numpy arrays
#   calls: state_dict, cpu, numpy | called by: Flower framework
#   process: extracts model parameters as list of numpy arrays for server

# FlowerClient.set_parameters - implemented, untested, unbackedup
#   input: parameters(list of numpy arrays) | output: None
#   calls: torch.tensor, load_state_dict | called by: Flower framework
#   process: loads server parameters into local model

# FlowerClient.fit - implemented, untested, unbackedup
#   input: parameters(list), config(dict) | output: tuple (parameters, num_samples, metrics_dict)
#   calls: set_parameters, train, get_parameters, compute_layer_norms, sanitize_update | called by: Flower framework
#   process: receives global model, trains locally, computes delta, sanitizes with ALDP-Dx, returns sanitized params

# FlowerClient.evaluate - implemented, untested, unbackedup
#   input: parameters(list), config(dict) | output: tuple (loss, num_samples, metrics_dict)
#   calls: set_parameters, evaluation loop | called by: Flower framework
#   process: evaluates global model on local test data, returns loss and accuracy

# train - implemented, untested, unbackedup
#   input: model(nn.Module), trainloader(DataLoader), epochs(int), device(str) | output: None
#   calls: optimizer.step, loss.backward | called by: FlowerClient.fit
#   process: standard PyTorch training loop for specified epochs

import torch
import torch.nn as nn
from collections import OrderedDict
from flwr.client import NumPyClient
from privacy import compute_layer_norms
from model import get_layer_groups


def train(model, trainloader, epochs, device):
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, privacy_engine, device="cpu"):
        self.model = model
        self.trainloader = trainloader
        self.privacy_engine = privacy_engine
        self.device = device
        self.layer_groups = get_layer_groups(model)
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        initial_params = [val.clone() for val in self.model.state_dict().values()]
        
        self.set_parameters(parameters)
        
        epochs = config.get("local_epochs", 1)
        train(self.model, self.trainloader, epochs, self.device)
        
        updated_params = [val.clone() for val in self.model.state_dict().values()]
        
        param_names = list(self.model.state_dict().keys())
        delta = OrderedDict()
        for i, name in enumerate(param_names):
            delta[name] = updated_params[i] - initial_params[i]
        
        norms = compute_layer_norms(delta, self.layer_groups)
        
        all_client_norms = config.get("all_client_norms", [norms])
        
        sanitized_delta = self.privacy_engine.sanitize_update(delta, all_client_norms)
        
        sanitized_params = []
        for i, name in enumerate(param_names):
            sanitized_params.append(initial_params[i] + sanitized_delta[name])
        
        sanitized_numpy = [p.cpu().numpy() for p in sanitized_params]
        
        return sanitized_numpy, len(self.trainloader.dataset), {"norms": norms}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        self.model.to(self.device)
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.trainloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        
        return loss, len(self.trainloader.dataset), {"accuracy": accuracy}