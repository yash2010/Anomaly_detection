from dataclasses import dataclass, field
from typing import List
import yaml

@dataclass
class dataConfig:
    file_path: str
    min_samples: int
    max_len: int 
    max_freq: int 

@dataclass
class modelConfig:
    filters: List[int]
    bottleneck_dim: int
    lr: float
    input_dim: int = None

@dataclass
class trainConfig:
    pretrain_epochs: int
    finetune_epochs: int
    batch_size: int
    beta_rep: float
    target_ratio: float
    update_centers: bool
    update_interval: int
    momentum: float

@dataclass
class clusterConfig:
    n_clusters: int

@dataclass
class automlConfig:
    n_trails: int
    bottleneck_dim: List[int]
    lr: float
    beta_rep: float
    n_clusters: int
    filters: List[List[int]]

@dataclass
class Config:
    data: dataConfig
    model: modelConfig
    train: trainConfig
    automl: automlConfig
    cluster: clusterConfig

def load_config(path:str) -> Config:
    with open (path) as f:
        raw = yaml.safe_load(f)
        data = dataConfig(**raw["data"])
        model = modelConfig(**raw["model"])
        model.input_dim = data.max_len
        return Config(
            data = data,
            model = model,
            train = trainConfig(**raw["train"]),
            cluster = clusterConfig(**raw["cluster"]),
            automl = automlConfig(**raw["automl"]),
        )

