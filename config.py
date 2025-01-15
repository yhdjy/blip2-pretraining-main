import torch
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Config:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    max_txt_len: int = 32
    epochs: int = 10
    lr: float = 0.002
    train_data_path: str = "data2/train_data.json"
    images_path: str = "data2/images"
    save_model_path: str = "output/model"  # 保存blip2模型地址


class oxford_pets_config:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size: int = 2
    max_txt_len: int = 32
    epochs: int = 10
    lr: float = 0.0005
    train_data_path: str = "DATA/oxford_pets/split_zhou_OxfordPets.json"
    images_path: str = "DATA/oxford_pets/images"
    save_model_path: str = "output/model"  # 保存blip2模型地址
    name: str = "oxford_pets"
    class_num: int = 19  # 基类19，新类18
    class_num_base: int = 19
    caption_num: int = 7
    new_class: bool = False


class food101_config:
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    batch_size: int = 2
    max_txt_len: int = 32
    epochs: int = 5
    lr: float = 0.0003
    train_data_path: str = "DATA/food-101/split_zhou_Food101.json"
    images_path: str = "DATA/food-101/images"
    save_model_path: str = "output/model2"  # 保存blip2模型地址
    name: str = "food-101"
    class_num: int = 51
    class_num_base: int = 51
    caption_num: int = 16
    new_class: bool = False


@dataclass
class Blip2QformerConfig:
    # Paths
    visual_encoder_model_path: str = "models/eva_vit_g.pth"
    qformer_model_path: str = "models/blip2_pretrained.pth"
    bert_base_uncased_path: str = "models/bert-base-uncased"

    # ViT encoder
    img_size: int = 224
    drop_path_rate: float = 0.0
    freeze_vit: bool = True

    # Q-Former
    num_query_token: int = 32


@dataclass
class ImageProcessorConfig:
    do_convert_rgb: bool = True
    do_normalize: bool = True
    do_rescale: bool = True
    do_resize: bool = True
    image_mean: List[float] = field(default_factory=lambda: [0.48145466, 0.4578275, 0.40821073])
    image_std: List[float] = field(default_factory=lambda: [0.26862954, 0.26130258, 0.27577711])
    resample: int = 3
    rescale_factor: float = 0.00392156862745098
    size: Dict[str, int] = field(default_factory=lambda: {"height": 224, "width": 224})