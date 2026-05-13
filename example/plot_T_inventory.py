"""Plot tritium inventory over time from simulation results."""
import json
import numpy as np
import matplotlib.pyplot as plt

with open("results_bin0_W_high_wetted/results.json") as f:
    data = json.load(f)

t = np.array(data["t"])
T_mobile = np.array(data["T"]["data"])
T_trap1 = np.array(data["trap1_T"]["data"])
T_trap2 = np.array(data["trap2_T"]["data"])
T_total = T_mobile + T_trap1 + T_trap2

fig, ax = plt.subplots(figsize=(10, 6))

# Primary axis: trapped T and total
ax.plot(t, T_trap1, label="T trap 1")
ax.plot(t, T_trap2, label="T trap 2")
ax.plot(t, T_total, linewidth=2, label="T total", color="black")
ax.set_xlabel("Time (s)")
ax.set_xscale("log")
ax.set_xlim(1e2, t[-1])
ax.set_ylabel("Trapped Inventory (atoms/m²)")
ax.set_title("Tritium inventory — Bin 0 (W, high_wetted, FW)")
ax.tick_params(axis="y", labelcolor="C0")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, t[-1])

# Secondary axis: mobile T
ax2 = ax.twinx()
ax2.plot(t, T_mobile, label="T mobile", color="gray", linestyle="--")
ax2.set_ylabel("Mobile Inventory (atoms/m²)", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()
plt.savefig("results_bin0_W_high_wetted/T_inventory.png", dpi=150)
print("Saved T_inventory.png")
plt.show()
