import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):      # 自定义类（继承pytorch.nn.Moudle）,将输入的 token 序列转换为相应的嵌入向量
    def __init__(self, config):  # 初始方法
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dimensions)
        # 使用nn.Embedding()定义一个嵌入层，num_embeddings 指定了嵌入层的词汇表大小（即可以嵌入的不同 token 的数量）。这里使用 config.vocab_size。
        # embedding_dim 指定了每个嵌入向量的维度。这里使用 config.embedding_dimensions

    def forward(self, tokenized_sentence):    # 定义一个实例方法（前向传播方法），接受一个token序列作为参数
        return self.token_embedding(tokenized_sentence)   # 将该参数通过刚刚定义的嵌入层进行处理，然后返回


class PositionalEncoding(nn.Module):   # 自定义类，用来实现位置编码（因为transformer里没有内置的位置信息），位置编码用于为序列中的每个位置添加位置信息
    def __init__(self, config):   # 初始方法，参数为先前定义的参数字典
        super().__init__()        # 调用父类的初始化方法，保证正确初始化父类（如果一个类有继承的父类，就要在初始方法内调用父类的初始方法）
        pe = torch.zeros(config.max_tokens, config.embedding_dimensions)   # 创建一个全零矩阵，形状为（max_tokens， embedding_dimensions）
        position = torch.arange(0, config.max_tokens, dtype=torch.float).unsqueeze(1)
        # 创建一个张量 position，包含从 0 到 config.max_tokens-1 的位置索引，并通过 unsqueeze(1) 将其形状从 [max_tokens] 变为 [max_tokens, 1]
        div_term = 1 / (
                    10000 ** (torch.arange(0, config.embedding_dimensions, 2).float() / config.embedding_dimensions))
        # 计算位置编码的缩放因子 div_term，用于缩放位置索引。这个缩放因子使得正弦和余弦函数的频率不同
        pe[:, 0::2] = torch.sin(position * div_term)  # 对 pe 的偶数列（即 0, 2, 4, ... 列）应用正弦函数，生成位置编码的正弦部分
        pe[:, 1::2] = torch.cos(position * div_term)  # 对 pe 的奇数列（即 1, 3, 5, ... 列）应用余弦函数，生成位置编码的余弦部分

        self.pe = pe.unsqueeze(0).transpose(0, 1)
        # 将 pe 张量的形状从 [max_tokens, embedding_dimensions] 变为 [1, max_tokens, embedding_dimensions]，
        # 然后再转置为 [max_tokens, 1, embedding_dimensions]，以便与输入数据的形状兼容
        self.pe = self.pe.to(config.device)   # 将位置编码矩阵 self.pe 移动到指定的设备（CPU 或 GPU）

    def forward(self, x):   # 定义前向传播方法
        return x + self.pe[:, 0]   # 返回输入张量与位置编码相加后的结果
        # self.pe 的形状是 [max_tokens, 1, embedding_dimensions]，
        # 我们使用 self.pe[:, 0] 提取出形状为 [max_tokens, embedding_dimensions] 的位置编码，然后与输入张量 x 相加


def scaled_dot_product_attention(query, key, value):  # 自定义函数，用于计算缩放点积注意力（scaled dot-product attention）。
    # 缩放点积注意力是 Transformer 模型中用于计算注意力权重和上下文向量的核心机制
    dim_k = query.size(-1)   # 获取 query 张量的最后一个维度的大小 dim_k，这是键/查询向量的维度，用于缩放分数
    scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(dim_k)
    # torch.bmm(query, key.transpose(1, 2))：计算 query 和 key 的点积
    # key.transpose(1, 2) 将 key 的形状从 [batch_size, seq_len_k, dim_k] 转置为 [batch_size, dim_k, seq_len_k]
    # 点积操作后，scores 的形状为 [batch_size, seq_len_q, seq_len_k]，表示每个查询与所有键之间的相似度
    # np.sqrt(dim_k)：点积结果除以 dim_k 的平方根进行缩放，防止随着 dim_k 增大，点积结果变得过大
    weights = F.softmax(scores, dim=-1)  # 对 scores 应用 softmax 函数，沿最后一个维度（即 seq_len_k 维度）计算权重，确保权重和为1
    # weights 的形状为 [batch_size, seq_len_q, seq_len_k]
    return torch.bmm(weights, value)    # 返回加权和
    # weights 的形状为 [batch_size, seq_len_q, seq_len_k]。
    # value 的形状为 [batch_size, seq_len_v, dim_v]。
    # 结果的形状为 [batch_size, seq_len_q, dim_v]，表示每个查询位置的上下文向量


class AttentionHead(nn.Module):     # 自定义类，用于实现单头注意力机制
    def __init__(self, embed_dim, head_dim):    # 初始方法，实现q、k、v的初始化，接受两个参数，embed_dim：输入嵌入向量的维度。head_dim：每个注意力头的维度。
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)   # 定义线性变换层 q，用于将输入嵌入向量转换为查询向量。输入维度为 embed_dim，输出维度为 head_dim
        self.k = nn.Linear(embed_dim, head_dim)   # 定义线性变换层 k，用于将输入嵌入向量转换为键向量。输入维度为 embed_dim，输出维度为 head_dim
        self.v = nn.Linear(embed_dim, head_dim)   # 定义线性变换层 v，用于将输入嵌入向量转换为值向量。输入维度为 embed_dim，输出维度为 head_dim

    def forward(self, hidden_state):   # 自定义方法（前向传播），接受一个参数hidden_state，即输入张量
        attn_outputs = scaled_dot_product_attention(self.q(hidden_state),   # 调用自定义函数，计算单头的点击缩放注意力，三个参数分别为将输入张量通过线性变换层转化的q、k、v
                                                    self.k(hidden_state),
                                                    self.v(hidden_state))
        return attn_outputs   # 返回单头的注意力输出


class MultiHeadAttention(nn.Module):   # 自定义类，实现多头注意力
    def __init__(self, config):   # 初始化方法
        super().__init__()
        embed_dim = config.embedding_dimensions  # 获取嵌入层维度
        num_heads = config.num_attention_heads   # 获取注意力头数
        head_dim = embed_dim // num_heads        # 获取每个注意力头的维度（嵌入层维度除以注意力头数）
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )    # 创建一个包含多个 AttentionHead 实例的模块列表 self.heads。每个 AttentionHead 实例接收 embed_dim 和 head_dim 作为参数
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        # 定义一个线性变换层 self.output_linear，用于将多头注意力的输出连接起来。输入和输出维度均为 embed_dim

    def forward(self, hidden_state):    # 实例方法（前向传播）
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        # 使用列表生成式 [h(hidden_state) for h in self.heads]，
        # 对每个 AttentionHead 实例 h 调用其前向传播方法，将 hidden_state 作为输入，得到每个注意力头的输出
        # torch.cat(..., dim=-1)：沿最后一个维度连接所有注意力头的输出，得到一个新的张量 x
        # 假设 hidden_state 的形状为 [batch_size, seq_len, embed_dim]，每个头的输出形状为 [batch_size, seq_len, head_dim]。
        # 连接后，x 的形状为 [batch_size, seq_len, embed_dim]，因为 embed_dim = num_heads * head_dim
        x = self.output_linear(x)  # 将连接后的张量 x 通过线性变换层 self.output_linear，得到最终的输出
        return x   # 返回多头注意力的最终输出


class FeedForward(nn.Module):    # 定义一个前馈神经网络
    # 前馈神经网络通常用于 Transformer 的每个编码器和解码器层中，以进一步处理和转换通过注意力机制得到的表示
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.embedding_dimensions, config.intermediate_size)
        # 定义一个线性投影层，用于将输入从嵌入维度 embedding_dimensions 转换为中间层维度 intermediate_size
        self.linear_2 = nn.Linear(config.intermediate_size, config.embedding_dimensions)
        # 定义第二个线性投影层，用于将中间维度重新转换为嵌入维度
        self.gelu = nn.GELU()   # 定义一个GELU激活函数
        self.dropout = nn.Dropout(config.hidden_dropout_prob)   # 定义一个 Dropout 层，用于在训练过程中随机丢弃一些神经元，以防止过拟合

    def forward(self, x):    # 自定义方法，接受一个参数为输入张量
        x = self.linear_1(x)   # 先经过第一个线性投影层，再经过激活函数，接着通过第二个线性投影层，最后经过droupt层
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x   # 返回处理完的张量


class PostLNEncoder(nn.Module):   # 自定义类，实现后层归一化（Post-Layer Normalization）架构
    # 实现了经典的 "Attention Is All You Need" 论文中的后层归一化（Post-Layer Normalization）架构
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)   # 实例化多头注意力模块
        self.layer_norm1 = nn.LayerNorm(config.embedding_dimensions)    # 初始化第一个层归一化模块
        self.layer_norm2 = nn.LayerNorm(config.embedding_dimensions)    # 初始化第二个层归一化模块
        self.feed_forward = FeedForward(config)       # 实例化前馈网络层

    def forward(self, x):    # 自定义方法（前向传播）
        # Layer normalization + skip connections over the self-attention block
        x = self.layer_norm1(x + self.attention(x))   # 将输入张量通过多头注意力模块处理后与原输入张量进行跳跃连接，然后通过第一个层归一化
        # Layer norm + skip connections over the FFN
        x = self.layer_norm2(x + self.feed_forward(x))   # 将上一步处理完的张量x通过前馈网络层处理后同样进行跳跃连接，然后通过第二个层归一化
        return x     # 返回处理完的张量x


class Encoder(nn.Module):   # 自定义类，实现了改进的前层归一化（Pre-Layer Normalization）架构。
    # 这个架构是基于 Transformer 模型的编码器模块的，预层归一化的变体通常在训练深层模型时表现更稳定
    "The improved pre-LN architecture"

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)      # 实例化多头注意力类
        self.layer_norm1 = nn.LayerNorm(config.embedding_dimensions)     # 初始化第一个层归一化
        self.layer_norm2 = nn.LayerNorm(config.embedding_dimensions)     # 初始化第二个层归一化
        self.feed_forward = FeedForward(config)          # 实例化前馈网络层

    def forward(self, x):
        # First perform layer normalization
        hidden_state = self.layer_norm1(x)     # 首先将输入张量通过第一个层归一化
        # Then apply attention + skip connection
        x = x + self.attention(hidden_state)   # 通过多头注意力后进行跳跃连接

        # Apply layer normalization before inputting to the FFN
        hidden_state = self.layer_norm2(x)     # 然后通过第二个层归一化
        # Apply FNN + skip connection
        x = x + self.feed_forward(hidden_state)   # 最后通过前馈网络层然后进行跳跃连接
        return x   # 返回处理完的张量


class ClassifierHead(nn.Module):    # 自定义类，实现二分类
    def __init__(self, config):
        super().__init__()
        self.flatten = nn.Flatten()     # 创建一个Flatten层，用于将多维数据展平为一维
        self.linear1 = nn.Linear(config.max_tokens * config.embedding_dimensions, 2 * config.embedding_dimensions)
        # 创建一个全连接层，输入维度为config.max_tokens * config.embedding_dimensions，映射维度为2 * config.embedding_dimensions
        self.relu = nn.ReLU()     # 定义激活函数为relu，用于引入非线性
        self.linear2 = nn.Linear(2 * config.embedding_dimensions, 1)
        # 定义第二个全连接层，将数据从2 * config.embedding_dimensions维度映射到单一的维度，用于输出最后的概率值

    def forward(self, x):    # 自定义方法（前向传播）
        x = self.flatten(x)    # 输入向量经过flatten层
        x = self.relu(self.linear1(x))   # 随后经过第一个线性投影层，然后通过激活函数
        x = self.linear2(x)    # 最后经过第二个线性投影层
        return torch.sigmoid(x)  # 对最后的输出进行归一化，即概率值


# transformer架构
class Transformer(nn.Module):   # 自定义类，实现transformer架构，进行二分类任务
    def __init__(self, config):    # 初始化方法，接收参数为模型字典对象
        super().__init__()
        self.embedding = TokenEmbedding(config)         # 实例化TokenEmbedding类
        self.positional_encoding = PositionalEncoding(config)    # 实例化位置编码类
        self.encoders = nn.ModuleList([Encoder(config) for _ in range(config.num_encoder_layers)])  # 创建两个编码器类，并实例化
        self.classifier_head = ClassifierHead(config)    # 实例化ClassifierHead类

    def forward(self, x):   # 前向传播，接受参数为输入张量
        x = self.embedding(x)   # 输入向量先经过TokenEmbedding层
        x = self.positional_encoding(x)   # 接着通过位置编码层
        for encoder in self.encoders:
            x = encoder(x)               # 然后通过两个编码层
        return self.classifier_head(x)   # 最后进行分类，返回预测结果
