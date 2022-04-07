import torch
import torch.nn as nn


class Co_Attention(nn.Module):
    def __init__(self, e_dim, grid, word_vocab_size, hashtag_vocab_size, k):
        super(Co_Attention, self).__init__()
        self.e_dim = e_dim
        self.grid = grid
        self.hashtag_vocab_size = hashtag_vocab_size
        self.k = k
        self.img_linear = nn.Linear(1000, e_dim)
        self.embedding = nn.Embedding(word_vocab_size, e_dim)
        self.feat_lstm = nn.LSTM(e_dim, e_dim, batch_first=True)
        self.W_vi = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(k, e_dim)))
        self.W_vt = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(k, e_dim)))
        self.W_pi = nn.Linear(2 * k, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.W_vih = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(k, e_dim)))
        self.W_t = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(k, e_dim)))
        self.W_pt = nn.Linear(2 * k, 1)
        self.fc = nn.Linear(e_dim, hashtag_vocab_size)

    def forward(self, img, txt):
        # feature 정리
        v_i = self.img_linear(img)  # [batch_size, grid, e_dim]
        txt = self.embedding(txt)  # [batch_size, maxlen, e_dim]
        maxlen = txt.shape[1]
        txt, (_, _) = self.feat_lstm(txt) # [batch_size, maxlen, e_dim]
        v_t = torch.mean(txt, dim=1).unsqueeze(-1)  # [batch_size, e_dim, 1]

        # tweet-guided visual attention
        h_i = torch.tanh(torch.concat((torch.einsum("ij,bjk->bik", self.W_vi, v_i.permute(0, 2, 1)),
                                       torch.einsum("ij,bjk->bik", self.W_vt, v_t).repeat(1, 1, self.grid)), dim=1))
        # [batch_size, 2k, grid]
        p_i = self.softmax(self.W_pi(h_i.permute(0, 2, 1)))  # [batch_size, grid, 1]
        vi_hat = torch.sum(p_i.repeat(1, 1, self.e_dim) * v_i, dim=1).unsqueeze(-1)
        # [batch_size, e_dim, 1]

        # image-guided textual attention

        h_t = torch.tanh(torch.concat((torch.einsum("ij,bjk->bik", self.W_vih, vi_hat.repeat(1, 1, maxlen)),
                                      torch.einsum("ij,bjk->bik", self.W_t, txt.permute(0,2,1))), dim=1))
        # [batch_size, 2k, maxlen]

        p_t = self.softmax(self.W_pt(h_t.permute(0, 2, 1)))  # [batch_size, maxlen, 1]
        # [batch_size, maxlen, 1]
        vt_hat = torch.sum(p_t.repeat(1, 1, self.e_dim) * txt, dim=1).unsqueeze(-1)
        # [batch_size, e_dim, 1]

        f = (vi_hat + vt_hat).squeeze()
        return self.fc(f) # [batch_size, hashtag_vocab_size]





