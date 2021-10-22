import torch

test_vals = torch.load("test_vals.pth")


import matplotlib.pyplot as plt

plt.plot(test_vals)
plt.show()