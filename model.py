import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=5000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Cannot use sin/cos positional encoding with odd dim (got dim={dim})")
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, dim)

        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, x, step=None):
        """
        Args:
            x: (B, T, D)
            step: optional step index (for autoregressive models)
        """
        x = x * math.sqrt(self.dim)
        if step is None:
            T = x.size(1)
            x = x + self.pe[:, :T]
        else:
            x = x + self.pe[:, step]
        return self.dropout(x)


class VideoEncoder(nn.Module):
    def __init__(self, embed_dim=2560, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, video_feat, video_mask=None):
        x = self.pos_encoding(video_feat)
        attn_mask = (video_mask == 0) if video_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=attn_mask)
        x = self.ln(x)
        return x   # (B, T, D)

class TextEncoder(nn.Module):
    def __init__(self, embed_dim=2560, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, text_feat, text_mask=None):
        B, C, T, D = text_feat.shape
        x = text_feat.view(B * C, T, D)
        mask = text_mask.view(B * C, T) if text_mask is not None else None
        attn_mask = (mask == 0) if mask is not None else None

        x = self.encoder(x, src_key_padding_mask=attn_mask)
        x = self.ln(x)
        return x  # (B*C, T, D)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
    

class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x_q, x_kv, mask_q=None, mask_kv=None):
        attn_mask = None
        if mask_kv is not None:
            attn_mask = (mask_kv == 0)
        x = self.ln(x_q)
        x_out, _ = self.cross_attn(query=x, key=x_kv, value=x_kv, key_padding_mask=attn_mask)
        x = x + x_out
        x = x + self.mlp(self.ln(x))
        return x
    

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x_q, x_kv, mask_q=None, mask_kv=None):
        for layer in self.layers:
            x_q = layer(x_q, x_kv, mask_q, mask_kv)
        return x_q

class VideoQAModel(nn.Module):
    def __init__(self, embed_dim=2560, hidden_dim=1024, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()

        self.video_input_proj = nn.Linear(embed_dim, hidden_dim)
        self.text_input_proj = nn.Linear(embed_dim, hidden_dim)

        self.video_encoder = VideoEncoder(hidden_dim, num_heads, num_layers, dropout)
        self.text_encoder = TextEncoder(hidden_dim, num_heads, num_layers, dropout)

        self.cross_attn_v2t = CrossAttentionBlock(hidden_dim, num_heads, num_layers, dropout)
        self.cross_attn_t2v = CrossAttentionBlock(hidden_dim, num_heads, num_layers, dropout)

        self.video_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            QuickGELU(),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6),
            QuickGELU(),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, video_feat, text_feat, video_mask=None, text_mask=None):
        B, C, T_text, D = text_feat.shape

        video_feat = self.video_input_proj(video_feat)  # [B, T_v, hidden_dim]
        text_feat = self.text_input_proj(text_feat)     # [B, C, T_text, hidden_dim]

        v_enc = self.video_encoder(video_feat, video_mask)
        t_enc = self.text_encoder(text_feat, text_mask)

        text_seq = t_enc.view(B * C, T_text, -1)
        v_seq = v_enc.unsqueeze(1).repeat(1, C, 1, 1).view(B * C, -1, v_enc.size(-1))

        t_mask = text_mask.view(B * C, T_text) if text_mask is not None else None
        v_mask = video_mask.unsqueeze(1).repeat(1, C, 1).view(B * C, -1) if video_mask is not None else None

        if t_mask is not None and v_mask is not None:
            invalid_t = (t_mask.sum(dim=1) == 0)
            if invalid_t.any():
                t_mask[invalid_t, 0] = 1

            invalid_v = (v_mask.sum(dim=1) == 0)
            if invalid_v.any():
                v_mask[invalid_v, 0] = 1

        v_cross = self.cross_attn_v2t(v_seq, text_seq, mask_q=v_mask, mask_kv=t_mask)
        t_cross = self.cross_attn_t2v(text_seq, v_seq, mask_q=t_mask, mask_kv=v_mask)

        v_pooled = v_cross.mean(dim=1).view(B, C, -1)
        t_pooled = t_cross.mean(dim=1).view(B, C, -1)

        v_proj = self.video_proj(v_pooled)
        t_proj = self.text_proj(t_pooled)

        v_emb = F.normalize(v_proj, dim=-1)
        t_emb = F.normalize(t_proj, dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = (v_emb * t_emb).sum(dim=-1) * logit_scale

        return logits