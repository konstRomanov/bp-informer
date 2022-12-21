import torch
import torch.nn as nn

from models.attn import FullAttention, AttentionLayer
from models.embed import DataEmbedding
from models.encoder import Encoder, EncoderLayer


class Informer(nn.Module):
    def __init__(self, ftr_num, d_out, pred_len,
                 d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, embed='fixed', freq='h', activation='gelu',
                 output_attention=False):
        super(Informer, self).__init__()
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(ftr_num, d_model, embed, freq, dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, d_out, bias=True)

    def forward(self, batch, batch_time, enc_self_mask=None):
        enc_in = self.enc_embedding(batch, batch_time)
        enc_out, attns = self.encoder(enc_in, attn_mask=enc_self_mask)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out[:, -self.pred_len:, :], attns
        else:
            return enc_out[:, -self.pred_len:, :]
