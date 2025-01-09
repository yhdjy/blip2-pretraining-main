import json
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, config, processor, tokenizer):
        self.config = config
        with open(self.config.train_data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)['train']  # 提取 "train" 数据
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        one = self.data[idx]
        image_path = f"{self.config.images_path}/{one[0]}"  # 使用文件名
        label = one[1]  # 标签
        breed = one[2]  # 品种名称（可选）

        # 加载图片
        image = Image.open(image_path).convert("RGB")  # 确保图片是 RGB 格式

        # 这里假设您有一些文本数据需要处理，您可以根据需要替换
        # 例如，您可以将 breed 作为文本输入
        text = breed

        # 处理文本
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_txt_len,
            return_tensors="pt"
        )

        # 处理输入数据
        inputs = self.processor(images=image, return_tensors="pt", padding=True)["pixel_values"].squeeze(0)

        return inputs, text_tokens