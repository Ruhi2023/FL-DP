# File contains: Entry point for ALDP-Dx federated learning simulation
# ** functions/classes
# client_fn - implemented, untested, unbackedup
#   input: cid(str) | output: FlowerClient
#   calls: get_resnet18, FlowerClient.__init__ | called by: Flower simulation
#   process: creates a client instance with model, data, privacy engine for given client ID

# main - implemented, untested, unbackedup
#   input: None | output: None
#   calls: get_resnet18, get_client_dataloaders, ALDPPrivacyEngine, ALDPStrategy, start_simulation | called by: script execution
#   process: sets up FL simulation with 3 clients, 3 rounds, runs end-to-end

import torch
from flwr.simulation import start_simulation
from flwr.server import ServerConfig

from model import get_resnet18
from data import get_client_dataloaders
from privacy import ALDPPrivacyEngine
from strategy import ALDPStrategy
from client import FlowerClient


def main():
    NUM_CLIENTS = 3
    NUM_ROUNDS = 3
    TOTAL_EPSILON = 1.0
    DELTA = 1e-5
    CLIP_NORM = 1.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")
    
    model = get_resnet18(num_classes=4)
    
    dataloaders = get_client_dataloaders(
        num_clients=NUM_CLIENTS,
        zip_path="brain_tumor_dataset.zip",  # Update with your actual zip filename
        batch_size=16
    )
    
    privacy_engine = ALDPPrivacyEngine(
        model=model,
        total_epsilon=TOTAL_EPSILON,
        delta=DELTA,
        clip_norm=CLIP_NORM
    )
    
    strategy = ALDPStrategy(
        privacy_engine=privacy_engine,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=0,
        min_available_clients=NUM_CLIENTS,
    )
    
    def client_fn(cid: str) -> FlowerClient:
        cid_int = int(cid)
        client_model = get_resnet18(num_classes=4)
        return FlowerClient(
            model=client_model,
            trainloader=dataloaders[cid_int],
            privacy_engine=privacy_engine,
            device=DEVICE
        )
    
    print(f"Starting FL simulation: {NUM_CLIENTS} clients, {NUM_ROUNDS} rounds")
    print(f"Privacy: ε={TOTAL_EPSILON}, δ={DELTA}, clip={CLIP_NORM}")
    
    history = start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )
    
    print("\n=== Simulation Complete ===")
    print(f"Completed {NUM_ROUNDS} rounds successfully")


if __name__ == "__main__":
    main()