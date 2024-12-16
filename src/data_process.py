import pandas as pd
import torch
import numpy as np

class data_process:
    def __init__(self, root_data_path):
        self.root_data_path = root_data_path
        # 设置显示选项，关闭科学计数法
        torch.set_printoptions(sci_mode=False)
        np.set_printoptions(suppress=True)
        pd.set_option('display.float_format', lambda x: '%.3f' % x)
        
    def process_data(self, rating_path, movie_path):
        # 读取数据
        rating_data = pd.read_csv(self.root_data_path + rating_path)
        movie_data = pd.read_csv(self.root_data_path + movie_path)
        
        # 获取唯一的用户ID和物品ID
        unique_users = rating_data['userId'].unique()
        unique_items = movie_data['movieId'].unique()
        
        # 创建ID到索引的映射
        user_id_to_index = {id_: idx for idx, id_ in enumerate(unique_users)}
        item_id_to_index = {id_: idx for idx, id_ in enumerate(unique_items)}
        
        # 创建索引到ID的映射
        index_to_user_id = {v: k for k, v in user_id_to_index.items()}
        index_to_item_id = {v: k for k, v in item_id_to_index.items()}
        
        # 转换用户ID和物品ID为索引
        user_indices = torch.tensor([user_id_to_index[id_] for id_ in rating_data['userId']], dtype=torch.long)
        item_indices = torch.tensor([item_id_to_index[id_] for id_ in rating_data['movieId']], dtype=torch.long)
        ratings = torch.tensor(rating_data['rating'].values, dtype=torch.float32)
        
        # 创建edge_index
        edge_index = torch.stack([
            item_indices,  # 第一行为项目的索引
            user_indices,  # 第二行为用户的索引
            ratings       # 第三行为评分
        ])
        
        # 创建映射字典
        id_mappings = {
            'user': {
                'id_to_index': user_id_to_index,
                'index_to_id': index_to_user_id
            },
            'item': {
                'id_to_index': item_id_to_index,
                'index_to_id': index_to_item_id
            }
        }
        
        # 获取用户和物品数量
        num_users = len(unique_users)
        num_items = len(unique_items)
        
        return edge_index, id_mappings, num_users, num_items