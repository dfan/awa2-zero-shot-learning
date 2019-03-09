import matplotlib.pyplot as plt
import numpy as np
import os
import re

# Plot loss from saved cluster output
f = open('output_train.txt', 'r')
losses = [float(num) for num in re.findall(r'Batch Loss: (.*)', f.read())]
iterations = [1 + 50*i for i in range(len(losses))]

# Plot loss and accuracy curves
plt.plot(iterations, losses, label='Binary Cross-Entropy Loss')
plt.yticks(np.arange(0, max(losses), 0.1))
plt.legend()
plt.xlabel("Iterations", fontweight='bold')
plt.ylabel("Training Batch Loss", fontweight='bold')

if not os.path.exists('figures'):
  os.mkdir('figures')
plt.savefig('figures/loss_plot.png', dpi=500)
plt.close()