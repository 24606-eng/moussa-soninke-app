# model.py — Tokeniseur SonikeTokenizer + Architecture GuppyLM
# Copié depuis le TP2. Ne pas modifier sauf si tu changes les hyperparamètres.

import torch
import torch.nn as nn
import math


class SonikeTokenizer:
    """Tokeniseur caractère-niveau adapté au soninké (alphabet SIL)."""
    def __init__(self):
        soninke_special = ['ŋ', 'ɲ', 'ɓ']
        chars = list("abcdefghijklmnopqrstuvwxyzàâäéèêëîïôùûüç .,!?;:'\"()-\n")
        chars += soninke_special
        chars = sorted(set(chars))
        self.special_tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.char2id = {**self.special_tokens}
        for i, c in enumerate(chars, start=len(self.special_tokens)):
            self.char2id[c] = i
        self.id2char    = {v: k for k, v in self.char2id.items()}
        self.vocab_size = len(self.char2id)

    def encode(self, text, max_len=128):
        ids = [self.special_tokens['<BOS>']]
        for c in text.lower():
            ids.append(self.char2id.get(c, self.special_tokens['<UNK>']))
        ids.append(self.special_tokens['<EOS>'])
        if len(ids) < max_len:
            ids += [self.special_tokens['<PAD>']] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids):
        chars = []
        for i in ids:
            if i in (self.special_tokens['<EOS>'], self.special_tokens['<PAD>']):
                break
            if i not in (self.special_tokens['<BOS>'],):
                chars.append(self.id2char.get(i, '?'))
        return ''.join(chars)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class GuppyLM(nn.Module):
    """Transformer encoder utilisé comme LM de dialogue soninké."""
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6,
                 d_ff=2048, max_len=128, dropout=0.1):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc    = PositionalEncoding(d_model, max_len, dropout)
        el = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(el, num_layers=n_layers)
        self.head        = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        pad_mask = (x == 0)
        x = self.pos_enc(self.embedding(x))
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.head(x)


def generate(model, tokenizer, prompt, max_new=80, temperature=0.8, device='cpu'):
    """Génère une réponse à partir d'un prompt soninké."""
    model.eval()
    ids = [i for i in tokenizer.encode(prompt, max_len=64) if i != 0][:64]
    x   = torch.tensor([ids], dtype=torch.long).to(device)
    generated = []
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(x)[0, -1, :] / temperature
            probs  = torch.softmax(logits, dim=-1)
            token  = torch.multinomial(probs, 1).item()
            if token == tokenizer.special_tokens['<EOS>']:
                break
            generated.append(token)
            x = torch.cat([x, torch.tensor([[token]]).to(device)], dim=1)
    return tokenizer.decode(generated)
