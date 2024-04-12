#!/usr/bin/env python
"""
PyTorch Implementation of Transformer from scratch

Reference:
github: https://github.com/hkproj/pytorch-transformer
youtube: https://www.youtube.com/watch?v=ISNdQcPhsts
"""
import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        @param d_model: embedding vector dimension/size
        @param vocab_size: vocabulary size
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        @param x: (batch_size, seq_len?) token indices
        @return:
        """
        # in the transformer paper, the embedding weights multiply sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        @param d_model: embedding vector dimension/size
        @param seq_len: sequence length
        @param dropout: dropout ratio
        """
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros([seq_len, d_model])
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        # Register the positional encoding as a buffer (save a non-learnable parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Embedding + Positional Encoding
        @param x: (batch_size, seq_len, d_model) - embedding vectors
        @return:
            x: (batch_size, seq_len, d_model)
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)  # (batch, seq_len, d_model)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        # y_i = gamma * (x_i - mu) / sqrt(sigma^2 + eps) + beta
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        """
        @param x: (batch_size, seq_len, n_channels)
        @return:
        """
        mu = x.mean(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        var = x.var(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        x = self.gamma * (x - mu) / torch.sqrt(var + self.eps) + self.beta
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        @param d_model: embedding dimension (512 in the paper)
        @param d_ff: inner-layer dimension (channel size) (2048 in the paper)
        @param dropout: dropout ratio
        @return:
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # W1 and b1 in the paper
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)  # W2 and b2 in the paper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        @param x: (batch_size, seq_len, d_model)
        @return:
            (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head attention for both the encoder and decoder
    """
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """
        @param d_model: embedding dimension
        @param h: number of heads
        @param dropout: dropout ratio
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h  # the dimension of each head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute attention.
        q, k, v are identical for the encoder. they are slightly different in the decoder.

        @param q: (batch_size, seq_len, d_model)
        @param k: (batch_size, seq_len, d_model)
        @param v: (batch_size, seq_len, d_model)
        @param mask: None for the encoder.
            In the decoder, we don't want the decoder to see the future words.
        @return:
            w_o(x): (batch_size, seq_len, d_model)
        """
        query = self.w_q(q)  # (batch_size, seq_len, d_model)
        key = self.w_k(k)  # (batch_size, seq_len, d_model)
        value = self.w_v(v)  # (batch_size, seq_len, d_model)

        # (batch_size, seq_len, d_model) --> (batch_size, h, seq_len, d_k)
        batch_size, seq_len = query.shape[0], query.shape[1]
        query = query.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)

        # calculate attention (x for next layer) and attention_scores (for visualization)
        x, attention_scores = self.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) --> (batch_size, seq_len, h, d_k)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.w_o(x)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """

        @param query: (batch_size, h, seq_len, d_k)
        @param key: (batch_size, h, seq_len, d_k)
        @param value: (batch_size, h, seq_len, d_k)
        @param mask: shape ?!!!
        @param dropout:
        @return:
            attention (attention_scores @ value): (batch_size, h, seq_len, d_k). For next layer.
            attention_scores: (batch_size, h, seq_len, seq_len). For visualization
        """
        d_k = query.shape[-1]
        # attention_scores:  (batch_size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # @ means torch.matmul
        # mask before applying softmax
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)  # replace with a very big negative number
        # attention_scores = nn.functional.softmax(attention_scores, dim=-1)  # (batch_size, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch_size, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value, attention_scores


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        """
        @param x:
        @param sublayer: a sub layer inside a residual block.
            It could be a FeedForwardBlock or a MultiHeadAttentionBlock.
        @return:
        """
        # the original paper: x + self.dropout(self.norm(sublayer(x)))
        # Most implementations apply nore before sublayer
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """
    A single Encoder Block contains two residual blocks
    """
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # the first sublayer is a multi-head attention layer
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, lambda x: self.feed_forward_block(x))
        return x


class Encoder(nn.Module):
    """ Nx of EncoderBlock """
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    """ A single decoder block """
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        # the first multi-head attention comes with a mask
        # the second multi-head attention is not self-attention, it's cross attention.
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """

        Encoder and decoder have different
        @param x:
        @param encoder_output:
        @param src_mask: encoder mask
        @param tgt_mask: decoder mask
        @return:
        """
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # for the cross_attention: key and value from the encoder
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(
            x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[3](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # note that the encoder_output is always the same for the Nx decoder blocks.
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """ A linear layer that projects the decoder output to the vocabulary """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        @param x: (batch_size, seq_len, d_model)
        @return:
            (batch_size, seq_len, vocab_size)
        """
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    """ Transformer network """
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbedding,
                 tgt_embed: InputEmbedding,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 proj_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.proj_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512,
                      N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    """

    @param src_vocab_size: source vocabulary size
    @param tgt_vocab_size: target vocabulary size
    @param src_seq_len: source sequence length
    @param tgt_seq_len: target sequence length (src_seq_len and tgt_seq_len can be the same or different)
    @param d_model: embedding dimension
    @param N: number of encoder/decoder blocks. (Nx, 6 in the paper).
    @param h: number ot multi-heads
    @param dropout: dropout ratio
    @param d_ff: inner-layer dimension in the FeedForwardBlock (2048 in the paper)
    @return:
    """
    # Create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    # Actually, src_pos and tgt_pos can be identical, and just use one positional encoding layer.
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder block
    encoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the project layer
    proj_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, proj_layer)

    # Initialize the parameters
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    return transformer


if __name__ == "__main__":
    torch.manual_seed(42)
    src_vocab_size = 25000
    tgt_vocab_size = 18000
    src_seq_len = 20
    tgt_seq_len = 20
    d_model = 512
    d_ff = 2048

    transformer = build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model)
    num_params = torch.sum(torch.tensor([param.numel() for param in transformer.parameters() if param.requires_grad]))
    print("Total number of learnable parameters: {}".format(num_params))

    batch_size = 4
    x = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))  # Example input tensor of token indices
    encoder_output = transformer.encode(x, None)
    print("Done")
