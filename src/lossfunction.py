import torch

def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, 
             neg_items_emb_final, neg_items_emb_0, edge_index, lambda_val=1e-6):
    """
    计算 BPR 损失
    """
    
    
    # 获取用户和物品的数量
    num_users = users_emb_final.shape[0]
    num_items = pos_items_emb_final.shape[0]
    
    # 确保索引在有效范围内
    user_indices = edge_index[1]
    item_indices = edge_index[0]
    
    # 检查索引是否有效
    assert torch.all(user_indices >= 0) and torch.all(user_indices < num_users), \
        f"User indices out of range. Max index: {user_indices.max()}, num_users: {num_users}"
    assert torch.all(item_indices >= 0) and torch.all(item_indices < num_items), \
        f"Item indices out of range. Max index: {item_indices.max()}, num_items: {num_items}"
    
    # 获取交互的用户和物品的嵌入
    interacted_users_emb = users_emb_final[user_indices]
    interacted_items_emb = pos_items_emb_final[item_indices]
    
    # 计算正样本得分
    pos_scores = torch.sum(interacted_users_emb * interacted_items_emb, dim=1)
    
    # 为每个交互生成有效的负样本索引
    neg_item_indices = torch.randint(0, num_items, (len(user_indices),), device=users_emb_final.device)
    neg_items_emb = neg_items_emb_final[neg_item_indices]
    
    # 计算负样本得分
    neg_scores = torch.sum(interacted_users_emb * neg_items_emb, dim=1)
    
    # 计算 BPR 损失
    bpr_loss_val = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
    
    # 计算正则化损失
    reg_loss = lambda_val * (
        torch.mean(users_emb_0[user_indices] ** 2) +
        torch.mean(pos_items_emb_0[item_indices] ** 2) +
        torch.mean(neg_items_emb_0[neg_item_indices] ** 2)
    )
    
    return bpr_loss_val + reg_loss, reg_loss, bpr_loss_val

