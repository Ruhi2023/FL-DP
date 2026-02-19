# File contains: ALDP-Dx privacy mechanisms - risk estimation, budget allocation, clipping, noise
# ** functions/classes
# compute_layer_norms - implemented, untested, unbackedup
#   input: model_delta(OrderedDict) | output: dict {layer_name: float}
#   calls: torch.norm | called by: ALDPPrivacyEngine.sanitize_update
#   process: computes L2 norm of all parameters within each layer group, returns dict of scalar norms

# compute_kl_risk - implemented, untested, unbackedup
#   input: current_norms(list of dict), reference_norms(list of dict) | output: dict {layer_name: float}
#   calls: scipy.stats.entropy | called by: ALDPPrivacyEngine.allocate_budget
#   process: builds histograms for P_l and Q_l from client norms, computes KL divergence per layer

# allocate_budget_inverse_softmax - implemented, untested, unbackedup
#   input: risk_dict(dict), total_epsilon(float) | output: dict {layer_name: float}
#   calls: np.exp, np.sum | called by: ALDPPrivacyEngine.allocate_budget
#   process: applies softmax(-R_l) then multiplies by total_epsilon to get per-layer budget

# ALDPPrivacyEngine - implemented, untested, unbackedup
#   input: model(nn.Module), total_epsilon(float), delta(float), clip_norm(float) | output: ALDPPrivacyEngine object
#   calls: get_layer_groups, compute_kl_risk, allocate_budget_inverse_softmax, compute_noise_multiplier | called by: client.py
#   process: stores layer groups, privacy params, history of norms; orchestrates risk computation and sanitization

# ALDPPrivacyEngine.store_round_norms - implemented, untested, unbackedup
#   input: norms_dict(dict) | output: None
#   calls: list.append | called by: strategy.py
#   process: appends current round's per-layer norms to history for next round's Q_l reference

# ALDPPrivacyEngine.allocate_budget - implemented, untested, unbackedup
#   input: current_norms(list of dict) | output: dict {layer_name: float}
#   calls: compute_kl_risk, allocate_budget_inverse_softmax | called by: ALDPPrivacyEngine.sanitize_update
#   process: if round 1 returns equal budget, else computes KL risk from history and allocates

# ALDPPrivacyEngine.sanitize_update - implemented, untested, unbackedup
#   input: model_delta(OrderedDict), current_round_all_client_norms(list of dict) | output: OrderedDict
#   calls: compute_layer_norms, allocate_budget, compute_noise_multiplier, clip_and_add_noise | called by: client.py
#   process: extracts layer norms, allocates budget, clips and adds noise per layer, returns sanitized delta

# clip_and_add_noise - implemented, untested, unbackedup
#   input: param_delta(Tensor), clip_value(float), noise_sigma(float) | output: Tensor
#   calls: torch.norm, torch.clamp, torch.randn_like | called by: ALDPPrivacyEngine.sanitize_update
#   process: clips gradient if norm exceeds clip_value, adds Gaussian noise scaled by noise_sigma

import torch
import numpy as np
from scipy.stats import entropy
from opacus.accountants.utils import get_noise_multiplier
from model import get_layer_groups


def compute_layer_norms(model_delta, layer_groups):
    norms = {}
    for layer_name, layer_module in layer_groups.items():
        layer_params = []
        for name, param in model_delta.items():
            if any(layer_name in name for layer_name in [layer_name]):
                layer_params.append(param.view(-1))
        if len(layer_params) > 0:
            concatenated = torch.cat(layer_params)
            norms[layer_name] = torch.norm(concatenated, p=2).item()
        else:
            norms[layer_name] = 0.0
    return norms


def compute_kl_risk(current_norms, reference_norms):
    risk = {}
    layer_names = current_norms[0].keys()
    
    for layer_name in layer_names:
        current_vals = [client_norms[layer_name] for client_norms in current_norms]
        ref_vals = [client_norms[layer_name] for client_norms in reference_norms]
        
        hist_current, bin_edges = np.histogram(current_vals, bins=10, density=True)
        hist_ref, _ = np.histogram(ref_vals, bins=bin_edges, density=True)
        
        hist_current = hist_current + 1e-10
        hist_ref = hist_ref + 1e-10
        
        kl_div = entropy(hist_current, hist_ref)
        risk[layer_name] = kl_div
    
    return risk


def allocate_budget_inverse_softmax(risk_dict, total_epsilon):
    layer_names = list(risk_dict.keys())
    risks = np.array([risk_dict[ln] for ln in layer_names])
    
    softmax_neg_risk = np.exp(-risks) / np.sum(np.exp(-risks))
    
    budget = {}
    for i, layer_name in enumerate(layer_names):
        budget[layer_name] = total_epsilon * softmax_neg_risk[i]
    
    return budget


class ALDPPrivacyEngine:
    def __init__(self, model, total_epsilon=1.0, delta=1e-5, clip_norm=1.0):
        self.layer_groups = get_layer_groups(model)
        self.total_epsilon = total_epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.norms_history = []
        
    def store_round_norms(self, norms_dict):
        self.norms_history.append(norms_dict)
        
    def allocate_budget(self, current_norms):
        if len(self.norms_history) == 0:
            num_layers = len(current_norms[0])
            equal_eps = self.total_epsilon / num_layers
            return {ln: equal_eps for ln in current_norms[0].keys()}
        
        risk = compute_kl_risk(current_norms, self.norms_history)
        return allocate_budget_inverse_softmax(risk, self.total_epsilon)
    
    def sanitize_update(self, model_delta, current_round_all_client_norms):
        this_client_norms = compute_layer_norms(model_delta, self.layer_groups)
        
        budget = self.allocate_budget(current_round_all_client_norms)
        
        sanitized_delta = {}
        
        for layer_name, epsilon_l in budget.items():
            layer_clip = self.clip_norm
            
            sigma_l = get_noise_multiplier(
                target_epsilon=epsilon_l,
                target_delta=self.delta,
                sample_rate=1.0,
                epochs=1
            )
            
            for param_name, param_delta in model_delta.items():
                if layer_name in param_name:
                    sanitized_delta[param_name] = clip_and_add_noise(
                        param_delta, layer_clip, sigma_l
                    )
        
        for param_name, param_delta in model_delta.items():
            if param_name not in sanitized_delta:
                sanitized_delta[param_name] = param_delta
        
        return sanitized_delta
    
    
def clip_and_add_noise(param_delta, clip_value, noise_sigma):
    norm = torch.norm(param_delta, p=2)
    clipped = param_delta * torch.clamp(clip_value / norm, max=1.0)
    noise = torch.randn_like(clipped) * noise_sigma * clip_value
    return clipped + noise