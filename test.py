from matplotlib import pyplot as plt

from src.components import PositionalEncoding

pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=1200)

p = pe.pe.squeeze(0)  # Remove the batch dimension

print(p.shape)

plt.imshow(p)

plt.title("Positional Encoding")
plt.show()