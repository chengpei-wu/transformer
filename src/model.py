from src.components import *


class Transformer(nn.Module):
    """
    Transformer model as described in "Attention is All You Need" paper.
    """
    def __init__(
            self,
            d_src_vocab,
            d_tgt_vocab,
            d_model=512,
            d_ff=2048,
            num_layer=6,
            num_head=8,
            dropout=0.1,
    ):
        """
        :param d_src_vocab: input source vocabulary size
        :param d_tgt_vocab: target vocabulary size
        :param d_model: feature dimension of the model
        :param d_ff: feed-forward hidden dimension
        :param num_layer: number of encoder/decoder layers
        :param num_head: number of attention heads
        :param dropout: dropout rate
        """
        super().__init__()
        self.src_embeding = Embedding(d_model, d_src_vocab)
        self.tgt_embeding = Embedding(d_model, d_tgt_vocab)
        self.d_model = d_model
        self.pe = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.encoder = Encoder(num_layer, d_model, num_head, d_model, d_ff)
        self.decoder = Decoder(num_layer, d_model, num_head, d_model, d_ff)
        self.generator = OutputGenerator(d_model=d_model, vocab=d_tgt_vocab)
        self.reset_param()

    def encode(self, x, src_mask):
        # encode
        pe_embedding = self.pe(self.src_embeding(x))
        encoder_output = self.encoder(pe_embedding, src_mask)
        return encoder_output

    def decode(self, x, memory, src_mask, tgt_mask):
        # decode
        pe_embedding = self.pe(self.tgt_embeding(x))
        decoder_output = self.decoder(pe_embedding, memory, src_mask, tgt_mask)
        return decoder_output

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.generator(
            self.decode(tgt, self.encode(src, src_mask), src_mask, tgt_mask)
        )

    def reset_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
