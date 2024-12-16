from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch
from torch_geometric.data import HeteroData

class KGAT(MessagePassing):
    def __init__(self, batch_size,embedding_dim,device):
        super(KGAT, self).__init__(aggr='sum', flow='source_to_target')
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.matrix_0 = nn.Parameter(torch.randn(embedding_dim, embedding_dim, requires_grad=True))
        self.matrix_1 = nn.Parameter(torch.randn(2*embedding_dim, 1, requires_grad=True))
        self.leakyrelu = nn.LeakyReLU(0.2)


    def forward(self, data: HeteroData):
        user_embedding = data["user"].x
        item_embedding = data["item"].x
        edge_index = data["edge_index"]
        edge_weight = data["edge_weight"]
        user_embedding = user_embedding.to(self.device)
        item_embedding = item_embedding.to(self.device)
        edge_index = edge_index.to(self.device).long()
        edge_weight = edge_weight.to(self.device)
        
        batch_size = self.batch_size
        #计算总样本数和每批大小
        total_nodes = len(user_embedding)
        user_embedding_out = self.graph_process(total_nodes,batch_size,user_embedding,
                                 item_embedding,edge_index,edge_weight)
        
        total_nodes = len(item_embedding)
        edge_index = torch.stack([edge_index[1], edge_index[0]])
        item_embedding_out = self.graph_process(total_nodes,batch_size,item_embedding,
                                 user_embedding,edge_index,edge_weight)
        return user_embedding_out,item_embedding_out
    
    
    def graph_process(self,total_nodes,batch_size,h_embedding,t_embedding,edge_index,edge_weight):
        # 存储所有批次的输出
        outputs = []
        # 按批次处理数据
        for start_idx in range(0, total_nodes, batch_size):
            end_idx = min(start_idx + batch_size, total_nodes)
        
            # 获取当前批次的节点
            batch_h_embedding = h_embedding[start_idx:end_idx]
            batch_t_embedding = t_embedding
            
            # 找出与当前批次相关的边
            mask = (edge_index[1] >= start_idx) & (edge_index[1] < end_idx)
            batch_edge_index = edge_index[:, mask]
            
            #batch_edge_weight = edge_weight[mask] if edge_weight is not None else None
            
            # 调整边索引以匹配批次内的局部索引
            batch_edge_index[1] = batch_edge_index[1] - start_idx
            
            # 处理当前批次
            batch_out = self.propagate(
                t=batch_t_embedding, 
                h=batch_h_embedding,
                edge_index=batch_edge_index
            )
            
            outputs.append(batch_out)
            
        
        # 合并所有批次的结果
        return torch.cat(outputs, dim=0)

    
    def message(self, t, h,edge_index):
        message_data = t[edge_index[0]]
        attention_score = self.attention(t, h, edge_index)
        
        message_data = message_data * attention_score
        #message_data = message_data * edge_weight
        return message_data
    
    def aggregate(self, inputs, edge_index):
        inputs = inputs.matmul(self.matrix_0)
        return super().aggregate(inputs, edge_index[1])
    
    def update(self, inputs, edge_index):
        return inputs
    
    def attention(self, t, h, edge_index):
        attention_score = torch.cat([h[edge_index[1]],t[edge_index[0]]],dim=1).to(self.device) #(message_size,2*embedding_dim)
        
        attention_score = attention_score.matmul(self.matrix_1) #(message_size,1)
        attention_score = self.leakyrelu(attention_score)
        #attention_score = torch.softmax(attention_score, dim=0)
        
        return attention_score

    def embedding(self, num_node):
        embedding_func = nn.Embedding(num_node, self.embedding_dim)
        embedding_func.weight.data.uniform_(-1, 1)
        embedding_data = torch.arange(num_node).long()
        embedding_data = embedding_func(embedding_data)

        return embedding_data
    
    







device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.ones(9, 10)
y = torch.ones(9, 10)
edge_index = torch.tensor([[0, 8, 2, 3, 4, 5, 6, 7, 8, 8,3], 
                           [1, 2, 3, 4, 5, 6, 7, 8, 1, 0,1]])
edge_weight = torch.ones(1, 11)
data = HeteroData()
data["user"].x = x
data["item"].x = y
data["edge_index"] = edge_index
data["edge_weight"] = edge_weight
data = data.to(device)
kgat = KGAT(5, 10,device).to(device)
out=kgat.forward(data=data)
print("out:",out)
