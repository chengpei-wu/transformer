import torch


def subsequent_mask(size):
    # attention mask for unseen words (tokens)
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


class Batch:
    # construct batch for training
    def __init__(self, src, tgt=None, pad=0, device=torch.device("cpu")):  # 2 = <blank>
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


def greedy_decode(model, src, src_mask, max_len, start_symbol, device):
    # output of encoder is used as memory for decoder
    memory = model.encode(src, src_mask)

    # initial input for decoder is the start symbol
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data).to(device)

    for i in range(max_len - 1):
        print('Input sequence:', ys.squeeze().cpu().numpy().tolist())

        out = model.decode(
            ys,
            memory,
            src_mask,
            subsequent_mask(ys.size(1)).type_as(src.data).to(device),
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]

        print('Predicted next word:', next_word.item())
        print()

        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


if __name__ == "__main__":
    print(subsequent_mask(10))
