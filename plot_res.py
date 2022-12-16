import numpy as np
import matplotlib.pyplot as plt

x_mae = np.array([50, 75, 90])
y_mae = np.array([71.746, 72.577, 70.52])

y_mae_no_reg = np.array([66.93, 66.762,65.951])

x_dmae = np.array([75, 90])
y_dmae = np.array([76.236, 75.267])



plt.plot(x_mae, y_mae, label="MAE full reg", marker = 'o', linestyle="dashed", c="tab:blue")
plt.plot(x_mae, y_mae_no_reg, label="MAE small reg", marker = 'o', linestyle="dashed", c="tab:orange")
plt.plot(x_dmae, y_dmae, label="DMAE", marker = 'o', linestyle="dashed", c="tab:red")

plt.xlabel("Masking ratio [%]")
plt.ylabel("Top-1 ACC [%]")
plt.title("Masking ratio ablation study")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("results/masking.png", dpi=300)
plt.show()
