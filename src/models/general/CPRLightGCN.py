from models.general.LightGCN import LightGCNBase
from models.BaseModel import BaseModel, GeneralModel
import numpy as np
import torch
import torch.nn as nn
import random

class CPRLightGCN(LightGCNBase, GeneralModel):
    reader = 'CPRReader'
    runner = 'CPRRunner'
    @staticmethod
    def parse_model_args(parser):
        parser = LightGCNBase.parse_model_args(parser)
        parser.add_argument('--beta', type=float, default=1.5,
							help='Dynamic sampling rate.')
        parser.add_argument('--gamma', type=int, default=2,
							help='The choosing rate.')
        parser.add_argument('--k', type=int, default=2,
							help='k .')
        parser.add_argument('--weight', type=float, default=0.001,
							help='正则化项系数 .')
        
        return GeneralModel.parse_model_args(parser)
    
    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self._base_init(args, corpus)
        self.k = args.k
        self.weight = args.weight
        self.beta = args.beta
        self.gamma = args.gamma
    
    def predict_before(self, feed_dict):
        return {'prediction': LightGCNBase.forward(self, feed_dict)['prediction']}
    
    def forward(self, feed_dict):
        user, items = feed_dict['user_id'], feed_dict['item_id']
        
        # 获取嵌入
        user_emb, item_emb = self.encoder(user, items)
        
        # 创建负样本嵌入
        neg_emb = torch.roll(item_emb, shifts=-1, dims=1)
        
        # 使用einsum计算预测分数
        pos_scores = torch.einsum('be,be->b', user_emb, item_emb)
        neg_scores = torch.einsum('be,be->b', user_emb, neg_emb)
        
        return {
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
            "batch_size": feed_dict["batch_size"]
        }

    def loss(self, out_dict):
        pos_scores = out_dict['pos_scores']  # [batch_size, num_samples] 假设形状
        neg_scores = out_dict['neg_scores']  # [batch_size, num_samples] 假设形状
        batch_size = out_dict['batch_size']

        # 计算 CPR 目标值
        if len(pos_scores.shape) > 1:
            cpr_obj = pos_scores.mean(dim=-1) - neg_scores.mean(dim=-1)
        else:
            cpr_obj = pos_scores - neg_scores  # 如果是标量或一维

        if cpr_obj.numel() > batch_size:
            k = min(batch_size, cpr_obj.numel())
            top_batch_size_cpr, _ = torch.topk(-cpr_obj, k=k)
        else:
            top_batch_size_cpr = -cpr_obj
        
        reg_loss = 0
        for param in self.parameters():
            reg_loss += torch.sum(param ** 2)
            reg_loss = self.weight * reg_loss
        
        # 计算 CPRLoss
        loss = torch.nn.Softplus()(top_batch_size_cpr).mean() + reg_loss

        return loss


    class Dataset(BaseModel.Dataset):
        def _get_feed_dict(self, index):
            user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
            if self.phase == 'train':
                item_ids = target_item
            else:
                if self.model.test_all:
                    neg_items = neg_items = np.arange(1, self.corpus.n_items)
                else:
                   neg_items = self.data['neg_items'][index]
                item_ids = np.concatenate([[target_item], neg_items]).astype(int)
            feed_dict = {
				'user_id': user_id,
				'item_id': item_ids
			}
            return feed_dict
        
        def dyn_collate_batch(self, feed_dicts):
            feed_dict = super().collate_batch(feed_dicts)
            batch_size = feed_dict['batch_size']
            num = int(batch_size * self.model.beta * self.model.gamma)
            
            user_item_pairs = [
                (d['user_id'], d['item_id'].item() if isinstance(d['item_id'], np.ndarray) and d['item_id'].size == 1 else d['item_id']) 
                for d in feed_dicts
            ]
            
            valid_samples = self._batch_sampling(user_item_pairs, num)
            feed_dict['user_id'], feed_dict['item_id'] = self._convert_to_tensors(valid_samples)
            return feed_dict

        def _batch_sampling(self, user_item_pairs, num):
            valid_samples = []
            while len(valid_samples) < num:
                batch_pairs = random.sample(user_item_pairs, min(self.model.k, len(user_item_pairs)))
                if self._validate_sample_group(batch_pairs):
                    valid_samples.append(batch_pairs)
            return valid_samples

        def _validate_sample_group(self, sample_group):
            """验证采样组的有效性"""
            for i, (user, _) in enumerate(sample_group):
                next_item = sample_group[(i + 1) % len(sample_group)][1]
                
                # 确保 next_item 是一个可哈希类型
                if isinstance(next_item, np.ndarray):
                    next_item = next_item.item() if next_item.size == 1 else tuple(next_item)
                
                if next_item in self.corpus.train_clicked_set[user]:
                    return False
            return True


        def _convert_to_tensors(self, valid_samples):
            users, items = zip(*[pair for sample_group in valid_samples for pair in sample_group])
            return torch.tensor(users), torch.tensor(items)
