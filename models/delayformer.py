import torch
from torch import nn
import math
from einops.layers.torch import Rearrange
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import TokenEmbedding, DataEmbedding_wo_pos, TemporalEmbedding, TimeFeatureEmbedding, PositionalEmbedding
from layers.RevIN import RevIN

class Model(nn.Module):
    def __init__(self, args):
        # if individual: input: [batch_size, seq_len, n_vars] → hankel: [batch_size, n_vars, h=seq_len-n+1, n=w] → 
        # patch: [batch_size*n_var, patch_num=h*w/p1/p2, p1*p2] → projection: [batch_size*n_var, patch_num=h*w/p1/p2, d_model] request: h%p1=w%p2=0
        # if channel-mixed: input: [batch_size, seq_len, n_vars] → hankel: [batch_size, n_vars, h=seq_len-n+1, n=w] → 
        # patch: [batch_size, patch_num=h*w/p1/p2, p1*p2*n_var]
        super(Model, self).__init__()
        self.task_name = args.task_name
        self.individual = args.individual
        self.d_model = args.d_model
        self.pe = args.pe
        self.temporal_encoding = args.temporal_encoding
        self.time_encoding = args.time_encoding
        self.project = args.project
        self.dropout = args.dropout
        self.n_heads = args.n_heads
        self.e_layers = args.e_layers
        self.d_ff = args.d_ff
        # self.d_ff = 4 * self.d_model
        
        # hankelize
        self.h = args.seq_len - args.L + 1
        self.w = args.L
        
        # patching
        self.n_vars = args.n_vars
        self.p1 = args.p1
        self.p2 = args.p2
        self.patch_num = self.h*self.w/self.p1/self.p2 #56
        if int(self.patch_num) == self.patch_num:
            self.patch_num = int(self.patch_num)
            pass
        else:
            raise RuntimeError('Please change hyper-paramters')
        
        self.patch_layer = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.p1, p2 = self.p2), # batch, token_num, token_dim 反向：Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=p, p2=p)
                                        )
        if self.individual:
            self.patch_size = self.p1*self.p2 # 42
        else:
            self.patch_size = self.p1*self.p2*self.n_vars #294
        
        if self.project == 'linear':
            self.W_projection = nn.Linear(self.patch_size, self.d_model)
        elif self.project == 'conv':
            self.W_projection = TokenEmbedding(self.patch_size, self.d_model)
        
        if self.pe == 'learnable_pe':
            self.position = positional_encoding('zeros', True, self.patch_num, self.d_model)
        elif self.pe == 'fix_pe':
            self.position = PositionalEmbedding(self.d_model)
        if self.temporal_encoding:
            self.time_embed = TemporalEmbedding(self.n_vars, args.embed, args.freq)
        if self.time_encoding:
            self.time_embed = TimeFeatureEmbedding(self.n_vars, args.embed, 'm')
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=self.dropout,
                                      output_attention=False), self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation='gelu'
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.pred_len = args.pred_len
        if self.task_name == 'imputation':
            self.pred_len = args.seq_len
        if self.task_name == 'anomaly_detection':
            self.pred_len = args.seq_len
            self.temporal_encoding = False
            self.time_encoding = False
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2, end_dim=-1))
                self.linears.append(nn.Linear(self.patch_num*self.d_model, self.pred_len))
                self.dropouts.append(nn.Dropout(self.dropout))
        else:
            self.pred_dim_change = nn.Sequential(nn.Linear(self.patch_num, self.n_vars),
                                                nn.LayerNorm(self.n_vars),
                                                nn.Dropout(self.dropout))
            self.mlp_head = nn.Sequential(nn.Linear(self.d_model, self.pred_len, bias=True),
                                          nn.Dropout(self.dropout))        
        self.revin_layer = RevIN(self.n_vars, affine=True, subtract_last=False)
    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        x = self.revin_layer(x, 'norm')
        if self.temporal_encoding or self.time_encoding:
            x = x + self.time_embed(x_mark_enc)
        x = hankel(x, self.w, True)
        if self.individual:
            tokens = [self.patch_layer(x[:,i,:,:].unsqueeze(1)) for i in range(x.shape[1])]
            tokens = torch.stack(tokens, dim=1)
            tokens = torch.reshape(tokens, (tokens.shape[0]*tokens.shape[1], tokens.shape[2], tokens.shape[3])) #(batch_size*n_var, patch_num, d_model)
        else:
            tokens = self.patch_layer(x)
            
        enc_in = self.W_projection(tokens)
        
        if self.pe == 'learnable_pos':
            enc_in = enc_in + self.position
        elif self.pe == 'fix_pos':
            enc_in = enc_in + self.position(enc_in)
        enc_out, _ = self.encoder(enc_in)
        if self.individual:
            enc_out = torch.reshape(enc_out, (-1, self.n_vars, enc_out.shape[-2], enc_out.shape[-1]))   #(batch_size, n_var, patch_num, d_model)
            z = []
            for i in range(self.n_vars):
                out = self.flattens[i](enc_out[:,i,:,:])
                out = self.linears[i](out)
                out = self.dropouts[i](out)
                z.append(out)
            z = torch.stack(z, dim=-1)
        else:
            z = self.pred_dim_change(enc_out.permute(0, 2, 1))
            z = self.mlp_head(z.permute(0, 2, 1)).permute(0, 2, 1)
        z = self.revin_layer(z, 'denorm')
        return z
    

def hankel(x, n, batch_data=False):
    if batch_data:
        batch_size, seq_len, variables = x.shape
        m = seq_len - n + 1
        idx = torch.arange(n).unsqueeze(0) + torch.arange(m).unsqueeze(1)
        hankels = x[:, idx, :]  # [batch_size, m, n, variables]
        return hankels.permute(0, 3, 1, 2)  # [batch_size, m, n, variables]
    else:
        seq_len, variables = x.shape
        m = seq_len - n + 1
        idx = torch.arange(n).unsqueeze(0) + torch.arange(m).unsqueeze(1)
        hankels = x[idx, :] # [m, n, variables]
        return hankels.permute(2, 0, 1)  # [variables, m, n]