import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(embed_dim, embed_dim)  # Query
        self.key_proj = nn.Linear(embed_dim, embed_dim)    # Key
        self.value_proj = nn.Linear(embed_dim, embed_dim)  # Value
        self.softmax = nn.Softmax(dim=-1)
        self.scale = embed_dim ** 0.5

    def forward(self, query, key, value):
        # Linear projections
        Q = self.query_proj(query)  # [Batch, Seq_Q, Embed_Dim]
        K = self.key_proj(key)      # [Batch, Seq_K, Embed_Dim]
        V = self.value_proj(value)  # [Batch, Seq_K, Embed_Dim]

        # Attention weights
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = self.softmax(attention_scores)

        # Weighted sum
        output = torch.matmul(attention_weights, V)
        return output



# 输入特征
main_feature = torch.randn(1, 256, 12, 8, 12).permute(0,2,3,4,1)      # 三维特征 [1, 256, 12, 8, 12]
horizontal_feature = torch.randn(1, 256, 12, 1, 12).permute(0,2,3,4,1)
expanded_horizontal_feature = horizontal_feature.expand(-1, -1, 8, -1, -1)  # 扩展为 [1, 12, 8, 12, 256]

# 使用 Cross-Attention 融合
cross_attention = CrossAttention(embed_dim=256)
fused_feature = cross_attention(expanded_horizontal_feature, main_feature, main_feature)
print(11)