import os

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.model import Transformer
from src.utils import Batch, greedy_decode


class CopyDataset(Dataset):
    def __init__(self, vocab_size, seq_len=10, total_samples=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.total_samples = total_samples

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 生成长度为 seq_len 的随机序列，第一个 token 强制为 1（即<sos>）
        data = torch.randint(1, self.vocab_size, size=(self.seq_len,))
        data[0] = 1
        return data.clone(), data.clone()  # 返回 src, tgt（完全一致）


def copy_task_collate_fn(batch, pad_token=0, device=torch.device("cpu")):
    # batch 是一个 list，元素是 (src, tgt) tuple
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch, dim=0)
    tgt_batch = torch.stack(tgt_batch, dim=0)
    return Batch(src_batch, tgt_batch, pad=pad_token, device=device)


def create_copy_task_dataloader(
        vocab_size,
        batch_size,
        total_samples=10000,
        device=torch.device("cpu")
):
    dataset = CopyDataset(vocab_size, total_samples=total_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: copy_task_collate_fn(batch, pad_token=0, device=device)
    )
    return dataloader


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, x, y, norm):
        sloss = (
                self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
                / norm
        )
        return sloss.data * norm, sloss


class LabelSmoothing(nn.Module):
    """
    Implement target label smoothing
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


if __name__ == "__main__":
    # Train the simple copy task for transformer.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # for mac
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")
    vocab_size = 11
    batch_size = 256

    criterion = LabelSmoothing(size=vocab_size, padding_idx=0, smoothing=0.0)
    model = Transformer(vocab_size, vocab_size, num_layer=2).to(device=device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embeding.d_model, factor=1.0, warmup=400
        ),
    )

    train_loader = create_copy_task_dataloader(vocab_size, batch_size, device=device, total_samples=batch_size * 10)
    val_loader = create_copy_task_dataloader(vocab_size, batch_size, device=device, total_samples=batch_size * 1)

    if not os.path.exists("./checkpoints/model_copy_task.pt"):
        with tqdm(desc="Training Progress", total=len(train_loader) * 30) as pbar:
            for epoch in range(30):
                for i, batch in enumerate(train_loader):
                    model.train()
                    optimizer.zero_grad()

                    output = model(
                        batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
                    )

                    _, loss = SimpleLossCompute(criterion)(
                        output, batch.tgt_y, norm=batch.ntokens
                    )
                    info = {
                        'Epoch': epoch + 1,
                        'Batch': i,
                        'Loss':  loss.item() / batch.ntokens.item(),
                    }

                    pbar.set_postfix(**info)
                    pbar.update(1)

                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

        torch.save(model.state_dict(), f"./checkpoints/model_copy_task.pt")
    else:
        model.load_state_dict(torch.load("./checkpoints/model_copy_task.pt", weights_only=True))

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]).to(device)
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len).to(device)
    greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0, device=device)
