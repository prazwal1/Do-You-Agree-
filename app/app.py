import json
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template

# --- PyTorch Model Definition from A4.ipynb ---

# Model hyperparameters (should match the trained model)
with open("models/tokenizer_metadata.json", "r") as f:
    meta = json.load(f)

vocab_size = meta["vocab_size"]
max_len = meta["max_len"]
d_model = meta["d_model"]
n_layers = meta["n_layers"]
n_heads = meta["n_heads"]
d_ff = meta["d_ff"]
d_k = meta["d_k"]
d_v = meta["d_v"]
n_segments = meta["n_segments"]
word2id = meta["word2id"]
pad_id = word2id["[PAD]"]

class Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    _, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(pad_id).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return self.norm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        # These parts are not needed for sentence embedding, but kept for loading state_dict
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def encode(self, input_ids, segment_ids):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, _ = layer(output, enc_self_attn_mask)
        return output

    def forward(self, input_ids, segment_ids, masked_pos=None):
        # Simplified forward for inference
        return self.encode(input_ids, segment_ids)

# --- Helper Functions ---

def clean_text(text):
    text = text.strip().lower()
    text = re.sub(r"[.,!?\\-]", "", text)
    return text

def encode_sentence(text, max_len, word2id):
    tokens = [word2id.get(w, word2id["[UNK]"]) for w in clean_text(text).split()]
    tokens = tokens[:max_len - 2]
    input_ids = [word2id["[CLS]"]] + tokens + [word2id["[SEP]"]]
    attention_mask = [1] * len(input_ids)
    pad_len = max_len - len(input_ids)
    input_ids.extend([pad_id] * pad_len)
    attention_mask.extend([0] * pad_len)
    return input_ids, attention_mask

def mean_pool(token_embeds, attention_mask):
    in_mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    pool = torch.sum(token_embeds * in_mask, dim=1) / torch.clamp(in_mask.sum(1), min=1e-9)
    return pool

# --- Flask App ---

app = Flask(__name__)

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sbert_checkpoint = torch.load("models/sbert_state_dict.pt", map_location=device)

model = BERT().to(device)
model.load_state_dict(sbert_checkpoint["bert_state_dict"])
model.eval()

classifier_head = nn.Linear(d_model * 3, 3).to(device)
classifier_head.load_state_dict(sbert_checkpoint["classifier_state_dict"])
classifier_head.eval()

label_map = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    premise = data.get('premise')
    hypothesis = data.get('hypothesis')

    if not premise or not hypothesis:
        return jsonify({'error': 'Premise and hypothesis are required.'}), 400

    # Prepare inputs
    premise_ids, premise_mask = encode_sentence(premise, max_len, word2id)
    hypothesis_ids, hypothesis_mask = encode_sentence(hypothesis, max_len, word2id)

    inputs_a = torch.tensor(premise_ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_a = torch.tensor(premise_mask, dtype=torch.long).unsqueeze(0).to(device)
    inputs_b = torch.tensor(hypothesis_ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_b = torch.tensor(hypothesis_mask, dtype=torch.long).unsqueeze(0).to(device)
    
    segment_ids_a = torch.zeros_like(inputs_a).to(device)
    segment_ids_b = torch.zeros_like(inputs_b).to(device)

    with torch.no_grad():
        u = model.encode(inputs_a, segment_ids_a)
        v = model.encode(inputs_b, segment_ids_b)

        u_mean = mean_pool(u, attention_a)
        v_mean = mean_pool(v, attention_b)

        uv_abs = torch.abs(u_mean - v_mean)
        x = torch.cat([u_mean, v_mean, uv_abs], dim=-1)
        logits = classifier_head(x)
        
        prediction_idx = torch.argmax(logits, dim=-1).item()
        prediction_label = label_map[prediction_idx]

    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
