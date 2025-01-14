import torch
import torch.nn as nn
from transformers import BertTokenizer
from .Qformer import BertConfig, BertLMHeadModel
from .standard_output import BlipOutput, LayerNorm
from torch.nn import functional as F
from .visiontransformer import VisionTransformer
from functools import partial
import numpy as np

class Blip2Qformer(nn.Module):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    """

    def __init__(
            self,
            img_size=224,
            drop_path_rate=0,
            freeze_vit=True,
            num_query_token=32,
            cross_attention_freq=2,
            embed_dim=256,
            max_txt_len=32,
            visual_encoder_model_path=None,
            qformer_model_path=None,
            bert_base_uncased_path=None,
            config=None,
            classname=None,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = self.init_tokenizer(bert_base_uncased_path)
        self.visual_encoder = self.init_vision_encoder(img_size, drop_path_rate)
        self.ln_vision = LayerNorm(self.visual_encoder.num_features)
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
        self.Qformer, self.query_tokens = self.init_qformer(
            num_query_token, self.visual_encoder.num_features, bert_base_uncased_path, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.load_model_weights(visual_encoder_model_path, qformer_model_path)

        self.adapter = SelfAttnAdapter(embed_dim, 4, ratio=0.5).to(self.config.device)
        #self.adapter = OfficialSelfAttentionLayer(embed_dim, 4)
        # todo 获取标签描述文本嵌入
        self.classname = classname
        #self.attr = self.get_attr2(classname)

        # gpt4_sentences = torch.load(f'./gpt4_data/{self.config.name}.pt')
        # # print('gpt4 sentences ', gpt4_sentences)
        #
        # self.attr2 = []
        # self.current_sentences = [None] * 100
        # self.tokenized_sentences = [None] * 100
        # self.current_input_ids = [None] * 100
        # self.current_attention_mask = [None] * 100
        # self.text_output = [None] * 100
        # self.text_feat = [None] * 100
        # i=0
        # # 获取所有类别名称的文本特征
        # for cl in classname:
        #     # 处理不同数据集的类别名称
        #     if self.config.name in ['OxfordFlowers', 'StanfordCars', 'EuroSAT']:
        #         pass
        #     else:
        #         cl = '_'.join(cl.split(' '))
        #
        #     self.current_sentences[i] = gpt4_sentences[cl.lower()]
        #     self.tokenized_sentences[i] = [self.tokenizer(
        #         c,
        #         padding="max_length",
        #         truncation=True,
        #         max_length=self.config.max_txt_len,
        #         return_tensors="pt",
        #     ) for c in self.current_sentences[i]]
        #
        #     # 将输入 ID 和注意力掩码移动到正确的设备
        #     self.current_input_ids[i] = [token["input_ids"].to(self.config.device) for token in self.tokenized_sentences[i]]
        #     self.current_attention_mask[i] = [token["attention_mask"].to(self.config.device) for token in self.tokenized_sentences[i]]
        #     self.Qformer = self.Qformer.to(self.config.device)
        #     # 合并张量并确保它们在同一设备上
        #     self.text_output[i] = self.Qformer.bert(
        #         torch.cat(self.current_input_ids[i], dim=0).to(self.config.device),  # 确保在正确的设备上
        #         attention_mask=torch.cat(self.current_attention_mask[i], dim=0).to(self.config.device),  # 确保在正确的设备上
        #         return_dict=True,
        #     )
        #     self.text_proj = self.text_proj.to(self.config.device)
        #     self.text_feat[i] = F.normalize(
        #         self.text_proj(self.text_output[i].last_hidden_state[:, 0, :]), dim=-1
        #     )
        #
        #     # 将 text_feat 移动到正确的设备
        #     self.attr2.append(self.text_feat[i].unsqueeze(0).to(self.config.device))
        #     i = i+1
        #
        # # 最终合并特征并确保在同一设备上
        # self.final_text_feats = torch.cat(self.attr2, dim=0)
        # self.attr = self.final_text_feats

    def init_vision_encoder(self, img_size, drop_path_rate):
        # 此处加载的是eva_clip_g模型， 加载需要改一下
        visual_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=14,
            use_mean_pooling=False,
            embed_dim=1408,
            depth=39,
            num_heads=1408 // 88,
            mlp_ratio=4.3637,
            qkv_bias=True,
            drop_path_rate=drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_checkpoint=False,
        )
        return visual_encoder

    def load_model_weights(self, vision_weights_path, qformer_model_path):
        # 加载视觉编码器权重
        vision_weights = torch.load(vision_weights_path, map_location="cpu")
        self.visual_encoder.load_state_dict(vision_weights, strict=False)
        print("Visual encoder weights loaded successfully!")

        # 加载其他权重
        rest_weights = torch.load(qformer_model_path, map_location="cpu")["model"]
        model_dict = self.state_dict()
        # 过滤掉视觉编码器的权重
        rest_weights = {k: v for k, v in rest_weights.items() if not k.startswith("visual_encoder")}
        model_dict.update(rest_weights)
        self.load_state_dict(model_dict)
        print("Remaining model weights loaded successfully!")

    def init_qformer(self, num_query_token, vision_width, bert_base_uncased_path, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(bert_base_uncased_path)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            bert_base_uncased_path, config=encoder_config
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def init_tokenizer(self, bert_base_uncased_path, truncation_side="right"):
        tokenizer = BertTokenizer.from_pretrained(bert_base_uncased_path,
                                                  truncation_side=truncation_side)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    # def get_attr(self, classnames):
    #
    #     gpt4_sentences = torch.load(f'./gpt4_data/oxford_pets.pt')
    #     print('gpt4 sentences ', gpt4_sentences)
    #
    #     attr = []
    #     # now get the text features for all the gpt4 sentences
    #     for cl in classnames:
    #         # need to include code for all datasets, some dont need the folowing line
    #         if self.config.name in ['OxfordFlowers', 'StanfordCars', 'EuroSAT']:
    #             pass
    #         else:
    #             cl = '_'.join(cl.split(' '))
    #         current_sentences = gpt4_sentences[cl.lower()]
    #         tokenized_sentences = [self.tokenizer(
    #             c,
    #             padding="max_length",
    #             truncation=True,
    #             max_length=self.config.max_txt_len,
    #             return_tensors="pt",
    #         ) for c in current_sentences]
    #         current_input_ids = [token["input_ids"] for token in tokenized_sentences]
    #         current_attention_masks = [token["attention_mask"] for token in tokenized_sentences]
    #         #todo 这里被init（）调用时是在cpu上计算
    #
    #         text_output = self.Qformer.bert(
    #             torch.cat(current_input_ids, dim=0),
    #             attention_mask=torch.cat(current_attention_masks, dim=0),
    #             return_dict=True,
    #         )
    #         text_feat = F.normalize(
    #             self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
    #         )
    #
    #         # current_sentences = current_sentences.to('cuda')
    #         attr.append(text_feat.unsqueeze(0))
    #     #attr = torch.stack(attr).to(self.device)
    #     final_text_feats = torch.cat(attr, dim=0)
    #     return final_text_feats

    def get_attr2(self, classnames):
        gpt4_sentences = torch.load(f'./gpt4_data/{self.config.name}.pt')
        #print('gpt4 sentences ', gpt4_sentences)

        attr = []
        # 获取所有类别名称的文本特征
        for cl in classnames:
            # 处理不同数据集的类别名称
            if self.config.name in ['OxfordFlowers', 'StanfordCars', 'EuroSAT']:
                pass
            else:
                cl = '_'.join(cl.split(' '))

            current_sentences = gpt4_sentences[cl.lower()]
            tokenized_sentences = [self.tokenizer(
                c,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_txt_len,
                return_tensors="pt",
            ) for c in current_sentences]

            # 将输入 ID 和注意力掩码移动到正确的设备
            current_input_ids = [token["input_ids"].to(self.config.device) for token in tokenized_sentences]
            current_attention_masks = [token["attention_mask"].to(self.config.device) for token in tokenized_sentences]
            self.Qformer = self.Qformer.to(self.config.device)
            # 合并张量并确保它们在同一设备上
            text_output = self.Qformer.bert(
                torch.cat(current_input_ids, dim=0).to(self.config.device),  # 确保在正确的设备上
                attention_mask=torch.cat(current_attention_masks, dim=0).to(self.config.device),  # 确保在正确的设备上
                return_dict=True,
            )
            self.text_proj = self.text_proj.to(self.config.device)
            text_feat = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )

            # 将 text_feat 移动到正确的设备
            attr.append(text_feat.unsqueeze(0).to(self.config.device))

        # 最终合并特征并确保在同一设备上
        final_text_feats = torch.cat(attr, dim=0)
        return final_text_feats
    def get_features(self, image, text_tokens):
        image_embeds = self.ln_vision(self.visual_encoder(image))  # 视觉encode, ln_vision是标准化
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # query_tokens 是原理图中的learned queries

        # 获得queries和图像融合的encode
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        # 获得文本encode
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        return image_feats,text_feat


    def forward(self, image, text_tokens):
        attr=self.get_attr2(self.classname)
        #attr = self.attr
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image=image.to(device)
        text_tokens=text_tokens.to(device)
        image_embeds = self.ln_vision(self.visual_encoder(image))  # 视觉encode, ln_vision是标准化
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # query_tokens 是原理图中的learned queries

        # 获得queries和图像融合的encode
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        # 获得文本encode
        # text_output = self.Qformer.bert(
        #     text_tokens.input_ids,
        #     attention_mask=text_tokens.attention_mask,
        #     return_dict=True,
        # )
        # text_feat = F.normalize(
        #     self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        # )

        #todo 将同一类的不同描述融合 [19,7,256]->[19,256]
        #attr=torch.rand(19,7,256).to(image_feats.device)
        #attr = attr.to(image_feats.device)
        text_features = self.adapter(attr)  # adapter前后都是[19,7,512]
        text_feat = text_features.mean(dim=1)  # [19,512]
        # 下面的loss 计算可以看博客https://zhuanlan.zhihu.com/p/16034558568
        ###============== Image-text Contrastive ===================###
        image_feats_all = image_feats  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = text_feat  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        # rank = dist.get_rank()
        rank = 0
        bs = image.size(0)
        # targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=torch.long).to(
        #     image.device
        # )
        targets = text_tokens
        #todo 暂时只计算图像到文本的损失
        # loss_itc = (F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) + F.cross_entropy(sim_t2i, targets,
        #                                                                                      label_smoothing=0.1)) / 2
        loss_itc = F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        return BlipOutput(
            loss=loss_itc.float() if isinstance(loss_itc, torch.Tensor) else None,
        )
        ###============== Image-text Matching ===================###
        text_input_ids_world = text_tokens.input_ids
        text_attention_mask_world = text_tokens.attention_mask
        image_embeds_world = image_embeds  # all_gather_with_grad(image_embeds)
        with torch.no_grad():

            sim_t2i[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)
            sim_i2t[:, rank * bs: rank * bs + bs].fill_diagonal_(-10000)

            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            image.device
        )

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss
        return BlipOutput(
            loss=(loss_itc + loss_itm + loss_lm).float() if isinstance(loss_itc, torch.Tensor) else None,
            loss_itc=loss_itc.float() if isinstance(loss_itc, torch.Tensor) else None,
            loss_itm=loss_itm.float() if isinstance(loss_itm, torch.Tensor) else None,
            loss_lm=loss_lm.float() if isinstance(loss_lm, torch.Tensor) else None,
        )

    @torch.no_grad()
    def evaluate(self, image, text_tokens):
        attr = self.get_attr2(self.classname)
        #attr = self.attr
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = image.to(device)
        text_tokens = text_tokens.to(device)
        image_embeds = self.ln_vision(self.visual_encoder(image))  # 视觉encode, ln_vision是标准化
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)  # query_tokens 是原理图中的learned queries

        # 获得queries和图像融合的encode
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        # 获得文本encode
        # text_output = self.Qformer.bert(
        #     text_tokens.input_ids,
        #     attention_mask=text_tokens.attention_mask,
        #     return_dict=True,
        # )
        # text_feat = F.normalize(
        #     self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        # )

        # todo 将同一类的不同描述融合 [19,7,256]->[19,256]
        # attr=torch.rand(19,7,256).to(image_feats.device)
        # attr = attr.to(image_feats.device)
        text_features = self.adapter(attr)  # adapter前后都是[19,7,512]
        text_feat = text_features.mean(dim=1)  # [19,512]
        # 下面的loss 计算可以看博客https://zhuanlan.zhihu.com/p/16034558568
        ###============== Image-text Contrastive ===================###
        image_feats_all = image_feats  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = text_feat  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        # rank = dist.get_rank()
        rank = 0
        bs = image.size(0)
        # targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=torch.long).to(
        #     image.device
        # )
        targets = text_tokens
        # todo 暂时只计算图像到文本的损失
        # loss_itc = (F.cross_entropy(sim_i2t, targets, label_smoothing=0.1) + F.cross_entropy(sim_t2i, targets,
        #                                                                                      label_smoothing=0.1)) / 2
        # 获取预测的类别
        # 确保 sim_i2t 是二维的
        if sim_i2t.dim() == 1:
            # 如果是 (num_classes,) 形状，增加一个维度
            sim_i2t = sim_i2t.unsqueeze(0)  # 变为 (1, num_classes)
        _, predicted = torch.max(sim_i2t, dim=1)  # 返回每行最大值的索引

        # 计算准确率
        correct_predictions = (predicted == targets).sum().item()
        total_predictions = targets.size(0)
        accuracy = correct_predictions / total_predictions

        print(f'Accuracy: {accuracy:.4f}')
        return correct_predictions,total_predictions

    @torch.no_grad()
    def generate(
            self,
            image,
            use_nucleus_sampling=False,
            num_beams=1,
            max_length=30,
            min_length=10,
            top_p=0.9,
            repetition_penalty=1.0,
    ):
        """
        Args:
            image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        # image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions


class SelfAttnAdapter(nn.Module):

    def __init__(self, c_in, reduction=4, ratio=0.5, device='cuda:0'):
        super(SelfAttnAdapter, self).__init__()
        self.attn = MultiHeadAttention(1, c_in,
            c_in//reduction, c_in//reduction, dropout=0.5, ratio=ratio).to(device)

    def forward(self, x):
        x = self.attn(x, x, x)
        return x


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v,
                 dropout=0.1, ratio=0.5):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        nn.init.xavier_normal_(self.w_qs.weight, gain=1.0)  # 自定义实现初始化
        nn.init.xavier_normal_(self.w_ks.weight, gain=1.0)
        nn.init.xavier_normal_(self.w_vs.weight, gain=0.67)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)  # feed forward layer
        nn.init.xavier_normal_(self.fc.weight, gain=0.67)

        self.dropout = nn.Dropout(dropout)

        self.ratio = ratio

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(2 * (self.ratio * output + (1 - self.ratio) * residual))  # 残差比例控制

        return output

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn



class OfficialSelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(OfficialSelfAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, x):
        attn_output, attn_weights = self.attention(x, x, x)
        return attn_output