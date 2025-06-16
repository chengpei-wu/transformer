import torch
from torch.utils.data import DataLoader, Dataset


def subsequent_mask(size):
    # attention mask for unseen words (tokens)
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


class Batch:
    # construct batch for training
    def __init__(self, src, tgt=None, pad=2, device=torch.device("cpu")):  # 2 = <blank>
        self.device = device

        # input for encoder
        self.src = src.to(device)

        # add padding mask for sentences that shorter than max_len
        self.src_mask = (src != pad).unsqueeze(-2).to(device)

        # add padding mask and future unseen
        if tgt is not None:
            # <sos> A B C <eos>

            # initial input for decoder: remove the last token
            # <sos> A B C
            self.tgt = tgt[:, :-1].to(device)

            # expected output for transformer (decoder): remove the first token
            # A B C <eos>
            self.tgt_y = tgt[:, 1:].to(device)

            # add masks for decoder input (both padding mask and future mask)
            self.tgt_mask = self.make_std_mask(self.tgt, pad)

            # the number of tokens in a batch
            self.ntokens = (self.tgt_y != pad).data.sum().to(device)

    def make_std_mask(self, tgt, pad):
        # Create a mask to hide padding and future words
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask.to(self.device)


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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 11  # toy example, tokens from 1 to 10 (0 is pad)
    batch_size = 2

    dataloader = create_copy_task_dataloader(vocab_size, batch_size, device=device)

    for batch in dataloader:
        print(batch.src)  # [batch_size, seq_len]
        print(batch.tgt)  # [batch_size, seq_len - 1]
        print(batch.tgt_y)  # [batch_size, seq_len - 1]
        print(batch.src_mask)  # [batch_size, 1, seq_len]
        print(batch.tgt_mask)  # [batch_size, seq_len - 1, seq_len - 1]
        break  # just check one batch
