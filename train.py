from model import Blip2Qformer
from transformers import BlipImageProcessor
from torch.utils.data import DataLoader
from config import Config, ImageProcessorConfig, Blip2QformerConfig
from load_data import CustomDataset
from torch.optim import Adam
import torch
from PIL import Image
import torch.nn.functional as F

class TrainBlip2:
    def __init__(self):
        blip2_qformer_config = Blip2QformerConfig().__dict__
        image_processor_config = ImageProcessorConfig().__dict__
        self.config = Config()
        self.blip2model = Blip2Qformer(**blip2_qformer_config).to(self.config.device)  # 加载blip2


        self.processor = BlipImageProcessor(**image_processor_config)  # 加载图像预处理

        dataset = CustomDataset(self.config, self.processor, self.blip2model.tokenizer)  # 读取数据
        self.dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        self.model_opt = Adam(self.blip2model.parameters(), lr=self.config.lr)  # 设置优化器

    def train_blip2(self):
        for epochs in range(self.config.epochs):
            for i, data in enumerate(self.dataloader):
                loss = self.blip2model(data[0], data[1])
                self.blip2model.zero_grad()
                loss.loss.backward()
                self.model_opt.step()
                print(loss)
                # self.save_model()
            # 是否保存模型
            #self.save_model

    def save_model(self):
        blip2_pretrained = self.blip2model.state_dict()
        # 移除视觉编码器部分的权重
        blip2_pretrained = {k: v for k, v in blip2_pretrained.items() if not k.startswith("visual_encoder")}
        torch.save({"model": blip2_pretrained}, f"{self.config.save_model_path}/blip2_pretrained.pth")

    def detect(self):
        image_path = "images/test.jpg"
        txt = ["a dog", "a bicycle", "a people", "a zebra"]
        # 加载图片
        image = Image.open(image_path)
        text_tokens = self.blip2model.tokenizer(
            txt,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_txt_len,
            return_tensors="pt",
        )  # .to(self.config.device)
        for key, val in text_tokens.items():
            text_tokens[key] = val.squeeze(0)

        # 处理输入数据
        inputs = self.processor(images=image, return_tensors="pt", padding=True)["pixel_values"].squeeze(0)
        new_tensor = inputs.unsqueeze(0).to(self.config.device)
        text_tokens.to(self.config.device)
        image_feats, text_feat = self.blip2model.get_features(new_tensor, text_tokens)

        image_feats_all = image_feats  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = text_feat  # [batch_size*num_gpu, embed_dim]


        sims_matrix = []
        for image_embed in image_feats_all:
            sim_q2t = image_embed @ text_feat_all.t()  # 1*32*1
            sim_i2t, _ = sim_q2t.max(0)
            probabilities = F.softmax(sim_i2t/0.05, dim=0)
            print(probabilities)




if __name__ == '__main__':
    train_blip2 = TrainBlip2()
    #train_blip2.detect()
    train_blip2.train_blip2()
