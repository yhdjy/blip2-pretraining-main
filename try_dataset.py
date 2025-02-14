from torch.optim.lr_scheduler import StepLR

from model import Blip2Qformer
from transformers import BlipImageProcessor
from torch.utils.data import DataLoader
from config import Config, ImageProcessorConfig, Blip2QformerConfig
from oxford_pets import CustomDataset
from oxford_pets import CustomDataset,CustomDataset_eva
from torch.optim import Adam
import torch
from PIL import Image
import torch.nn.functional as F
from config import oxford_pets_config,food101_config,dtd_config,ucf101_config,eurosat_config
#from peft import get_peft_model, LoraConfig
class TrainBlip2:
    def __init__(self):

        blip2_qformer_config = Blip2QformerConfig().__dict__
        image_processor_config = ImageProcessorConfig().__dict__
        self.config = food101_config()
        token = torch.rand(35, 7, 32)
        classname = self.get_classname()
        self.blip2model = Blip2Qformer(config=self.config, classname=classname, **blip2_qformer_config).to(self.config.device)  # 加载blip2
        #self.blip2model.attr = self.blip2model.get_attr2(classname)
        torch.save(self.blip2model.get_attr2(classname), "attrs.pt")
        self.blip2model.attr = torch.load("attrs.pt")
        # 打印所有参数名和对应的形状
        for name, param in self.blip2model.named_parameters():
            print(f"Parameter Name: {name}, Shape: {param.shape}")

        # pretrained_data = torch.load(f"{self.config.save_model_path}/blip2_pretrained.pth", map_location=self.config.device)
        # self.blip2model.load_state_dict(pretrained_data["model"], strict=False)
        # 冻结模型的所有参数
        for param in self.blip2model.parameters():
            param.requires_grad = False

        # 解冻适配器的参数
        for name, param in self.blip2model.named_parameters():
            print(name)
            if 'adapter' in name or 'cross_attn' in name or 'batch_norm' in name:# or 'encoder.layer.11' in name:  # 假设适配器的参数名称中包含 'adapter'
                param.requires_grad = True
        self.processor = BlipImageProcessor(**image_processor_config)  # 加载图像预处理
        self.log_file = "training_log.txt"  # 日志文件名
        # # 定义LoRA配置
        # lora_config = LoraConfig(
        #     r=8,  # 低秩矩阵的秩
        #     lora_alpha=16,  # LoRA的缩放因子
        #     target_modules=["layer.11.attention.self.query","layer.11.attention.self.key", ]  # 需要应用LoRA的模块
        # )
        # # 将LoRA配置应用到模型
        # self.blip2model = get_peft_model(self.blip2model, lora_config)
        dataset = CustomDataset(self.config, self.processor, self.blip2model.tokenizer)  # 读取数据

        #self.attr = self.blip2model.get_attr(classnames=classname)
        self.dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        self.model_opt = Adam(self.blip2model.parameters(), lr=self.config.lr)  # 设置优化器

        self.scheduler = StepLR(self.model_opt, step_size=self.config.step_size, gamma=self.config.gamma)  # 学习率调度器

    def log(self, message):
        with open(self.log_file, "a") as f:  # 以追加模式打开文件
            f.write(message + "\n")
    def train_blip2(self):
        #self.evaluate()

        for epochs in range(self.config.epochs):

            for i, data in enumerate(self.dataloader):
                loss = self.blip2model(data[0], data[1])
                if (i + 1) % 20 == 0:
                    self.blip2model.evaluate(data[0], data[1])
                self.blip2model.zero_grad()
                loss.loss.backward()
                self.model_opt.step()
                if (i + 1) % 20 == 0:
                    print(loss)
                if (i + 1) % 20 == 0:  # 每 10 个 batch 打印一次
                    print(f"  Batch [{i + 1}/{len(self.dataloader)}]: Loss = {loss.loss.item():.4f}")
                # self.save_model()
            print(f"epoch:{epochs}")
            acc = self.evaluate()
            learning_rate = self.model_opt.param_groups[0]['lr']
            info = f"epoch:{epochs}, acc:{acc:.4f}, lr: {learning_rate:.6f}  "
            self.log(info)
            self.scheduler.step()

            # 是否保存模型
        self.save_model(epochs)
        acc = self.evaluate()


    def save_model(self,epoch):
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

    def evaluate(self):
        dataset = CustomDataset_eva(self.config, self.processor, self.blip2model.tokenizer)  # 读取数据

        #self.attr = self.blip2model.get_attr(classnames=classname)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        correct_predictions = 0
        total_predictions = 0
        for i, data in enumerate(dataloader):

            a,b = self.blip2model.evaluate(data[0], data[1])
            correct_predictions+=a
            total_predictions+=b
            # 计算并打印进度百分比
            progress = (i + 1) / len(dataloader) * 100  # 计算进度
            print(f'Progress: {progress:.2f}% ({i + 1}/{len(dataloader)})')  # 打印进度
            print(f'total_acc: {correct_predictions*100/total_predictions:.4f}% ')  # 打印进度
        print(correct_predictions/total_predictions)
        return correct_predictions/total_predictions

    def get_classname(self):
        classname1 = [
            'abyssinian',
            'american_bulldog',
            'american_pit_bull_terrier',
            'basset_hound',
            'beagle',
            'bengal',
            'birman',
            'bombay',
            'boxer',
            'british_shorthair',
            'chihuahua',
            'egyptian_mau',
            'english_cocker_spaniel',
            'english_setter',
            'german_shorthaired',
            'great_pyrenees',
            'havanese',
            'japanese_chin',
            'keeshond'
        ]
        classname1_n = [
            'leonberger',
            'maine_coon',
            'miniature_pinscher',
            'newfoundland',
            'persian',
            'pomeranian',
            'pug',
            'ragdoll',
            'russian_blue',
            'saint_bernard',
            'samoyed',
            'scottish_terrier',
            'shiba_inu',
            'siamese',
            'sphynx',
            'staffordshire_bull_terrier',
            'wheaten_terrier',
            'yorkshire_terrier',
        ]
        classname2 = [
            'apple_pie',
            'baby_back_ribs',
            'baklava',
            'beef_carpaccio',
            'beef_tartare',
            'beet_salad',
            'beignets',
            'bibimbap',
            'bread_pudding',
            'breakfast_burrito',
            'bruschetta',
            'caesar_salad',
            'cannoli',
            'caprese_salad',
            'carrot_cake',
            'ceviche',
            'cheese_plate',
            'cheesecake',
            'chicken_curry',
            'chicken_quesadilla',
            'chicken_wings',
            'chocolate_cake',
            'chocolate_mousse',
            'churros',
            'clam_chowder',
            'club_sandwich',
            'crab_cakes',
            'creme_brulee',
            'croque_madame',
            'cup_cakes',
            'deviled_eggs',
            'donuts',
            'dumplings',
            'edamame',
            'eggs_benedict',
            'escargots',
            'falafel',
            'filet_mignon',
            'fish_and_chips',
            'foie_gras',
            'french_fries',
            'french_onion_soup',
            'french_toast',
            'fried_calamari',
            'fried_rice',
            'frozen_yogurt',
            'garlic_bread',
            'gnocchi',
            'greek_salad',
            'grilled_cheese_sandwich',
            'grilled_salmon',

        ]
        classname3 = [
            'banded',
            'blotchy',
            'braided',
            'bubbly',
            'bumpy',
            'chequered',
            'cobwebbed',
            'cracked',
            'crosshatched',
            'crystalline',
            'dotted',
            'fibrous',
            'flecked',
            'freckled',
            'frilly',
            'gauzy',
            'grid',
            'grooved',
            'honeycombed',
            'interlaced',
            'knitted',
            'lacelike',
            'lined',
            'marbled',

        ]
        classname4 = [
            'Apply_Eye_Makeup',
            'Apply_Lipstick',
            'Archery',
            'Baby_Crawling',
            'Balance_Beam',
            'Band_Marching',
            'Baseball_Pitch',
            'Basketball',
            'Basketball_Dunk',
            'Bench_Press',
            'Biking',
            'Billiards',
            'Blow_Dry_Hair',
            'Blowing_Candles',
            'Body_Weight_Squats',
            'Bowling',
            'Boxing_Punching_Bag',
            'Boxing_Speed_Bag',
            'Breast_Stroke',
            'Brushing_Teeth',
            'Clean_And_Jerk',
            'Cliff_Diving',
            'Cricket_Bowling',
            'Cricket_Shot',
            'Cutting_In_Kitchen',
            'Diving',
            'Drumming',
            'Fencing',
            'Field_Hockey_Penalty',
            'Floor_Gymnastics',
            'Frisbee_Catch',
            'Front_Crawl',
            'Golf_Swing',
            'Haircut',
            'Hammer_Throw',
            'Hammering',
            'Handstand_Pushups',
            'Handstand_Walking',
            'Head_Massage',
            'High_Jump',
            'Horse_Race',
            'Horse_Riding',
            'Hula_Hoop',
            'Ice_Dancing',
            'Javelin_Throw',
            'Juggling_Balls',
            'Jump_Rope',
            'Jumping_Jack',
            'Kayaking',
            'Knitting',
            'Long_Jump',

        ]
        classname5 = [
            'annual crop land',
            'forest',
            'herbaceous vegetation land',
            'highway or road',
            'industrial buildings',

        ]
        if self.config.name == 'oxford_pets':
            if not self.config.new_class:
                return classname1
            else:
                return classname1_n
        if self.config.name == 'food-101':
            return classname2
        if self.config.name == 'dtd':
            return classname3
        if self.config.name == 'ucf101':
            return classname4
        if self.config.name == 'eurosat':
            return classname5
        return classname1


if __name__ == '__main__':
    train_blip2 = TrainBlip2()
    #train_blip2.detect()
    train_blip2.train_blip2()