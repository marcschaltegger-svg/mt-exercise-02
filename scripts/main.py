# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
from model import PositionalEncoding, RNNModel, TransformerModel

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--accel', action='store_true',
                    help='Enables accelerated training')
parser.add_argument('--use-optimizer', action='store_true',
                    help='Uses AdamW optimizer for gradient updating')
#log-file flag
parser.add_argument('--log-file', type=str, default='',
                    help='Path to a TSV file for saving per-epoch train/val perplexities.')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

if args.accel and torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = torch.device("cpu")

print("Using device:", device)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss()
if args.use_optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

###############################################################################
# Log-file helpers
###############################################################################

def init_log_file(path: str) -> None:
    """Create (or overwrite) the TSV log file and write the header row."""
    log_dir = os.path.dirname(path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(path, 'w') as f:
        # Columns: epoch | train_ppl | val_ppl | dropout | model_type
        f.write('epoch\ttrain_ppl\tval_ppl\tdropout\tmodel\n')

def append_log(path: str, epoch: int, train_ppl: float, val_ppl: float) -> None:
    """Append one data row to the TSV log file."""
    with open(path, 'a') as f:
        f.write(f'{epoch}\t{train_ppl:.4f}\t{val_ppl:.4f}\t{args.dropout}\t{args.model}\n')

if args.log_file:
    init_log_file(args.log_file)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(data_source):
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train():
    """Run one full training epoch; returns the mean per-token loss."""
    model.train()
    total_loss = 0.
    epoch_total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    num_batches = 0
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        if args.use_optimizer:
            optimizer.zero_grad()
        else:
            model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if args.use_optimizer:
            optimizer.step()
        else:
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()
        epoch_total_loss += loss.item()
        num_batches += 1

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

    # Return mean per-token loss for the whole epoch (used for logging)
    return epoch_total_loss / num_batches if num_batches > 0 else float('inf')


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train()          # now returns mean epoch loss
        val_loss = evaluate(val_data)

        train_ppl = math.exp(train_loss)
        val_ppl   = math.exp(val_loss)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, val_ppl))
        print('-' * 89)

        # --- NEW: write epoch perplexities to log file ---
        if args.log_file:
            append_log(args.log_file, epoch, train_ppl, val_ppl)

        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    if args.model == 'Transformer':
        safe_globals = [
            PositionalEncoding,
            TransformerModel,
            torch.nn.functional.relu,
            torch.nn.modules.activation.MultiheadAttention,
            torch.nn.modules.container.ModuleList,
            torch.nn.modules.dropout.Dropout,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
            torch.nn.modules.normalization.LayerNorm,
            torch.nn.modules.sparse.Embedding,
            torch.nn.modules.transformer.TransformerEncoder,
            torch.nn.modules.transformer.TransformerEncoderLayer,
        ]
    else:
        safe_globals = [
            RNNModel,
            torch.nn.modules.dropout.Dropout,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.rnn.GRU,
            torch.nn.modules.rnn.LSTM,
            torch.nn.modules.rnn.RNN,
            torch.nn.modules.sparse.Embedding,
        ]
    with torch.serialization.safe_globals(safe_globals):
        model = torch.load(f)
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

if len(args.onnx_export) > 0:
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
