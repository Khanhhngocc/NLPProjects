import random
import math
import time
from collections import Counter

import numpy as np
import spacy

import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)

spacy_de = spacy.blank("de")
spacy_en = spacy.blank("en")

def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)][::-1]

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]


raw_datasets = load_dataset("bentrevett/multi30k")

print(raw_datasets)
print(raw_datasets["train"][0])

def preprocess_example(example):
    return {
        "src_tokens": tokenize_de(example["de"]),
        "trg_tokens": tokenize_en(example["en"])
    }

tokenized_datasets = raw_datasets.map(
    preprocess_example,
    remove_columns=raw_datasets["train"].column_names
)

print(tokenized_datasets["train"][0])

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [
    UNK_TOKEN,
    PAD_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
]

class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.itos = []
        self.stoi = {}

    def build(self, token_sequences):
        counter = Counter()

        for tokens in token_sequences:
            counter.update(tokens)

        self.itos = list(SPECIAL_TOKENS)
        already_added = set(self.itos)

        for token, freq in counter.items():
            if freq >= self.min_freq and token not in already_added:
                self.itos.append(token)
                already_added.add(token)

        self.stoi = {
            token: index
            for index, token in enumerate(self.itos)
        }

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        # Token chưa có trong vocab -> <unk>
        return self.stoi.get(token, self.stoi[UNK_TOKEN])

    def encode(self, tokens):
        return [self[token] for token in tokens]

    def decode(self, indices):
        return [self.itos[index] for index in indices]

src_vocab = Vocabulary(min_freq=2)
trg_vocab = Vocabulary(min_freq=2)

src_vocab.build(tokenized_datasets["train"]["src_tokens"])
trg_vocab.build(tokenized_datasets["train"]["trg_tokens"])

print(f"Source vocabulary: {len(src_vocab):,}")
print(f"Target vocabulary: {len(trg_vocab):,}")


SRC_PAD_IDX = src_vocab[PAD_TOKEN]
TRG_PAD_IDX = trg_vocab[PAD_TOKEN]

SRC_SOS_IDX = src_vocab[SOS_TOKEN]
SRC_EOS_IDX = src_vocab[EOS_TOKEN]

TRG_SOS_IDX = trg_vocab[SOS_TOKEN]
TRG_EOS_IDX = trg_vocab[EOS_TOKEN]

def numericalize(tokens, vocab):
    ids = [
        vocab[SOS_TOKEN],
        *vocab.encode(tokens),
        vocab[EOS_TOKEN],
    ]

    return torch.tensor(ids, dtype=torch.long)

def collate_batch(batch):
    src_batch = []
    trg_batch = []

    for example in batch:
        src_tensor = numericalize(
            example["src_tokens"],
            src_vocab,
        )

        trg_tensor = numericalize(
            example["trg_tokens"],
            trg_vocab,
        )

        src_batch.append(src_tensor)
        trg_batch.append(trg_tensor)

    # batch_first=False:
    # [sequence_length, batch_size]
    src_padded = pad_sequence(
        src_batch,
        batch_first=False,
        padding_value=SRC_PAD_IDX,
    )

    trg_padded = pad_sequence(
        trg_batch,
        batch_first=False,
        padding_value=TRG_PAD_IDX,
    )

    return src_padded, trg_padded

BATCH_SIZE = 128

train_iterator = DataLoader(
    tokenized_datasets["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch,
    num_workers=0,
)

valid_iterator = DataLoader(
    tokenized_datasets["validation"],
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
    num_workers=0,
)

test_iterator = DataLoader(
    tokenized_datasets["test"],
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
    num_workers=0,
)

src, trg = next(iter(train_iterator))

print("src shape:", src.shape)
print("trg shape:", trg.shape)



class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(
            input_dim,
            emb_dim,
            padding_idx=SRC_PAD_IDX,
        )

        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=1,
            bidirectional=True,
        )

        self.fc_hidden = nn.Linear(
            hid_dim * 2,
            hid_dim,
        )

        self.fc_cell = nn.Linear(
            hid_dim * 2,
            hid_dim,
        )

        self.dropout = nn.Dropout(dropout)



    def forward(self, src):
        # src = [src_len, batch_size]

        embedded = self.dropout(
            self.embedding(src)
        )
        # embedded = [src_len, batch_size, emb_dim]

        encoder_states, (hidden, cell) = self.rnn(
            embedded
        )

        # encoder_states = [src_len, batch_size, 2 * hid_dim]
        # hidden         = [2, batch_size, hid_dim]
        # cell           = [2, batch_size, hid_dim]

        hidden = self.fc_hidden(
            torch.cat(
                (hidden[0:1], hidden[1:2]),
                dim=2,
            )
        )

        cell = self.fc_cell(
            torch.cat(
                (cell[0:1], cell[1:2]),
                dim=2,
            )
        )

        # hidden = [1, batch_size, hid_dim]
        # cell   = [1, batch_size, hid_dim]

        return encoder_states, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim

        self.embedding = nn.Embedding(
            output_dim,
            emb_dim,
            padding_idx=TRG_PAD_IDX,
        )

        self.rnn = nn.LSTM(
            hid_dim * 2 + emb_dim,
            hid_dim,
            num_layers=1,
        )

        self.energy = nn.Linear(
            hid_dim * 3,
            1,
        )

        self.fc_out = nn.Linear(
            hid_dim,
            output_dim,
        )

        self.softmax = nn.Softmax(dim=0)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            input_token,
            encoder_states,
            hidden,
            cell,
    ):
        # input_token    = [batch_size]
        # encoder_states = [src_len, batch_size, 2 * hid_dim]
        # hidden         = [1, batch_size, hid_dim]
        # cell           = [1, batch_size, hid_dim]

        input_token = input_token.unsqueeze(0)

        # [1, batch_size]
        embedded = self.dropout(
            self.embedding(input_token)
        )
        # [1, batch_size, emb_dim]

        src_len = encoder_states.shape[0]

        h_reshaped = hidden.repeat(
            src_len,
            1,
            1,
        )

        # [src_len, batch_size, hid_dim]

        energy = self.relu(
            self.energy(
                torch.cat(
                    (h_reshaped, encoder_states),
                    dim=2,
                )
            )
        )

        # energy = [src_len, batch_size, 1]

        attention = self.softmax(energy)

        # attention = [src_len, batch_size, 1]

        attention_bmm = attention.permute(
            1,
            2,
            0,
        )
        # attention_bmm = [batch_size, 1, src_len]

        encoder_states_bmm = encoder_states.permute(
            1,
            0,
            2,
        )
        # encoder_states_bmm = [batch_size, src_len, hid_dim * 2]

        context_vector = torch.bmm(
            attention_bmm,
            encoder_states_bmm,
        )

        # [batch_size, 1, 2 * hid_dim]

        rnn_input = torch.cat(
            (context_vector, embedded),
            dim=2,
        )

        output, (hidden, cell) = self.rnn(
            rnn_input,
            (hidden, cell),
        )

        prediction = self.fc_out(
            output.squeeze(0)
        )
        # prediction = [batch_size, output_dim]

        return prediction, hidden, cell



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, (
            "Hidden dimensions of encoder and decoder must be equal!"
        )

    def forward(
        self,
        src,
        trg,
        teacher_forcing_ratio=0.5,
    ):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]

        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(
            trg_len,
            batch_size,
            trg_vocab_size,
            device=self.device,
        )

        encoder_states, hidden, cell = self.encoder(src)

        # trg[0] là <sos>
        input_token = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(
                input_token,
                encoder_states,
                hidden,
                cell,
            )

            outputs[t] = output

            teacher_force = (
                random.random()
                < teacher_forcing_ratio
            )

            top1 = output.argmax(dim=1)

            input_token = (
                trg[t]
                if teacher_force
                else top1
            )

        return outputs


INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(trg_vocab)

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512

ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(
    INPUT_DIM,
    ENC_EMB_DIM,
    HID_DIM,
    ENC_DROPOUT,
)

dec = Decoder(
    OUTPUT_DIM,
    DEC_EMB_DIM,
    HID_DIM,
    DEC_DROPOUT,
)

model = Seq2Seq(
    enc,
    dec,
    device,
).to(device)


def init_weights(m):
    for _, param in m.named_parameters():
        nn.init.uniform_(
            param.data,
            -0.08,
            0.08,
        )


model.apply(init_weights)

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


print(
    f"The model has {count_parameters(model):,} "
    "trainable parameters"
)


optimizer = optim.Adam(
    model.parameters()
)

criterion = nn.CrossEntropyLoss(
    ignore_index=TRG_PAD_IDX
)


def train(
    model,
    iterator,
    optimizer,
    criterion,
    clip,
):
    model.train()

    epoch_loss = 0

    for src, trg in iterator:
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        # output = [trg_len, batch_size, output_dim]
        # trg    = [trg_len, batch_size]

        output_dim = output.shape[-1]

        # Bỏ vị trí đầu tiên vì đó là <sos>.
        output = output[1:].reshape(
            -1,
            output_dim,
        )

        trg = trg[1:].reshape(-1)

        loss = criterion(
            output,
            trg,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            clip,
        )

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(
    model,
    iterator,
    criterion,
):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for src, trg in iterator:
            src = src.to(device)
            trg = trg.to(device)

            output = model(
                src,
                trg,
                teacher_forcing_ratio=0,
            )

            output_dim = output.shape[-1]

            output = output[1:].reshape(
                -1,
                output_dim,
            )

            trg = trg[1:].reshape(-1)

            loss = criterion(
                output,
                trg,
            )

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time

    elapsed_mins = int(
        elapsed_time / 60
    )

    elapsed_secs = int(
        elapsed_time
        - elapsed_mins * 60
    )

    return elapsed_mins, elapsed_secs

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float("inf")

MODEL_PATH = "seq2seq-attention-lstm-modern.pt"

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(
        model,
        train_iterator,
        optimizer,
        criterion,
        CLIP,
    )

    valid_loss = evaluate(
        model,
        valid_iterator,
        criterion,
    )

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(
        start_time,
        end_time,
    )

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH,
        )

    print(
        f"Epoch: {epoch + 1:02} | "
        f"Time: {epoch_mins}m {epoch_secs}s"
    )

    print(
        f"\tTrain Loss: {train_loss:.3f} | "
        f"Train PPL: {math.exp(train_loss):7.3f}"
    )

    print(
        f"\t Val. Loss: {valid_loss:.3f} | "
        f" Val. PPL: {math.exp(valid_loss):7.3f}"
    )


state_dict = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(state_dict)

test_loss = evaluate(
    model,
    test_iterator,
    criterion,
)

print(
    f"| Test Loss: {test_loss:.3f} | "
    f"Test PPL: {math.exp(test_loss):7.3f} |"
)
