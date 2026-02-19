# File contains: Custom Flower strategy for ALDP-Dx coordination
# ** functions/classes
# ALDPStrategy - implemented, untested, unbackedup
#   input: privacy_engine(ALDPPrivacyEngine), other FedAvg params | output: ALDPStrategy object
#   calls: FedAvg methods, configure_fit, aggregate_fit | called by: main.py
#   process: extends FedAvg to collect client norms, distribute them, store history

# ALDPStrategy.configure_fit - implemented, untested, unbackedup
#   input: server_round(int), parameters(Parameters), client_manager(ClientManager) | output: list of tuples (client, fit_config)
#   calls: super().configure_fit | called by: Flower framework
#   process: adds all_client_norms from previous round to config for each client

# ALDPStrategy.aggregate_fit - implemented, untested, unbackedup
#   input: server_round(int), results(list), failures(list) | output: tuple (parameters, metrics)
#   calls: super().aggregate_fit, privacy_engine.store_round_norms | called by: Flower framework
#   process: extracts norms from client results, stores them, aggregates parameters using FedAvg

from flwr.server.strategy import FedAvg
from flwr.common import Parameters


class ALDPStrategy(FedAvg):
    def __init__(self, privacy_engine, **kwargs):
        super().__init__(**kwargs)
        self.privacy_engine = privacy_engine
        self.current_round_norms = []
        
    def configure_fit(self, server_round, parameters, client_manager):
        config = {"local_epochs": 1}
        
        if len(self.current_round_norms) > 0:
            config["all_client_norms"] = self.current_round_norms
        
        fit_configs = super().configure_fit(server_round, parameters, client_manager)
        
        updated_configs = []
        for client, base_config in fit_configs:
            merged_config = {**base_config, **config}
            updated_configs.append((client, merged_config))
        
        return updated_configs
    
    def aggregate_fit(self, server_round, results, failures):
        norms_this_round = []
        
        for client_proxy, fit_res in results:
            if "norms" in fit_res.metrics:
                norms_this_round.append(fit_res.metrics["norms"])
        
        if len(norms_this_round) > 0:
            for norms_dict in norms_this_round:
                self.privacy_engine.store_round_norms(norms_dict)
            
            self.current_round_norms = norms_this_round
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        return aggregated_parameters, aggregated_metrics