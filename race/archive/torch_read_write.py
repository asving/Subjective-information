"""Do write directions settle before read directions? A measurement on a real small transformer.

Needs PyTorch (the other scripts in this directory do not).

Model: one layer, width 32, two attention heads, 64 ReLU neurons, learned position embeddings, no
layer norm. All weight matrices start as Gaussians of standard deviation 0.02. Trained with Adam
(learning rate 3e-3) on the full set of 256 aligned RRXOR sequences of length 12.

For each MLP neuron, the read is its row of the input matrix and the write is its column of the
output matrix. For the ten neurons that end up largest, the script reports
  - the step at which their writes, and their reads, come within cosine 0.9 of their final directions;
  - how closely the final writes line up with the logit direction (the difference of the two
    unembedding rows), and how much of the writes' and reads' variance lies in a single direction;
  - for seed 0, the loss when either all reads or all writes are put back to their initial values.

Run:  python torch_read_write.py     (under a minute)
"""
import itertools
import math

import torch
import torch.nn.functional as F

torch.set_num_threads(2)
L, D, H, M = 12, 32, 2, 64
STEPS = 1500

free = [p for p in range(1, L + 1) if p % 3]
seqs = []
for bits in itertools.product((0, 1), repeat=len(free)):
    s, it = [], iter(bits)
    for p in range(1, L + 1):
        s.append(s[-1] ^ s[-2] if p % 3 == 0 else next(it))
    seqs.append(s)
X = torch.tensor(seqs)

READS = ('win', 'bin', 'Q', 'K', 'V', 'U')      # what each component takes from the stream (U is the final reader)
WRITES = ('wout', 'O', 'tok', 'pos')            # what each component puts into the stream


def make(seed, std=0.02):
    torch.manual_seed(seed)
    shapes = dict(tok=(2, D), pos=(L, D), Q=(D, D), K=(D, D), V=(D, D), O=(D, D), win=(M, D), wout=(D, M), U=(2, D))
    P = {k: (torch.randn(*shape) * std).requires_grad_() for k, shape in shapes.items()}
    P['bin'] = torch.zeros(M, requires_grad=True)
    return P


def loss_of(P):
    x = X[:, :-1]
    B, T = x.shape
    r = P['tok'][x] + P['pos'][:T]
    q = (r @ P['Q'].t()).view(B, T, H, -1).transpose(1, 2)
    k = (r @ P['K'].t()).view(B, T, H, -1).transpose(1, 2)
    v = (r @ P['V'].t()).view(B, T, H, -1).transpose(1, 2)
    att = q @ k.transpose(-1, -2) / math.sqrt(D // H)
    att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float('-inf')).softmax(-1)
    r = r + (att @ v).transpose(1, 2).reshape(B, T, D) @ P['O'].t()
    a = F.relu(r @ P['win'].t() + P['bin'])
    logits = (r + a @ P['wout'].t()) @ P['U'].t()
    return F.cross_entropy(logits.reshape(-1, 2), X[:, 1:].reshape(-1)) / math.log(2)


def cos(a, b):
    return F.cosine_similarity(a, b, dim=-1)


def main():
    print(f"aligned RRXOR, width {D}, {M} neurons, Adam, {STEPS} steps; baseline loss 1.000, optimum 0.636")
    print("seed | loss<0.95 at | writes settle | reads settle | final loss | |cos(write, logit dir)| | variance in one direction: writes, reads")
    for seed in range(5):
        P = make(seed)
        init = {k: v.detach().clone() for k, v in P.items()}
        opt = torch.optim.Adam(P.values(), 3e-3)
        hist, resets = [], []
        for step in range(STEPS + 1):
            opt.zero_grad()
            ls = loss_of(P)
            ls.backward()
            hist.append((ls.item(), P['wout'].detach().t().clone(), P['win'].detach().clone()))
            if seed == 0 and step in (0, 30, 60, 90, 150, 300, STEPS):
                with torch.no_grad():
                    row = [step, ls.item()]
                    for group in (READS, WRITES):
                        Q = {k: (init[k] if k in group else v) for k, v in P.items()}
                        row.append(loss_of(Q).item())
                    resets.append(row)
            opt.step()
        fw, fr = hist[-1][1], hist[-1][2]
        top = (fw.norm(dim=1) * fr.norm(dim=1)).topk(10).indices
        t_loss = next(i for i, h in enumerate(hist) if h[0] < 0.95)
        t_w = next(i for i, h in enumerate(hist) if cos(h[1][top], fw[top]).mean() > 0.9)
        t_r = next(i for i, h in enumerate(hist) if cos(h[2][top], fr[top]).mean() > 0.9)
        logit_dir = (P['U'][1] - P['U'][0]).detach()
        align = cos(fw[top], logit_dir.expand_as(fw[top])).abs().mean().item()
        sw, sr = torch.linalg.svdvals(fw[top]), torch.linalg.svdvals(fr[top])
        print(f"{seed:4d} | {t_loss:12d} | {t_w:13d} | {t_r:12d} | {hist[-1][0]:10.4f} | {align:22.2f} |"
              f" {(sw[0] ** 2 / (sw ** 2).sum()).item():.2f}, {(sr[0] ** 2 / (sr ** 2).sum()).item():.2f}")
        if seed == 0:
            keep = resets
    print("\nseed 0: loss with one group put back to its initial values")
    print("  step   loss   reads reset   writes reset")
    for step, ls, lr_, lw_ in keep:
        print(f"{step:6d}  {ls:.4f}     {lr_:.4f}        {lw_:.4f}")


if __name__ == '__main__':
    main()
