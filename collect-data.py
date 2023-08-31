from gatherstat import GatherStatistics
import matplotlib.pyplot as plt
import numpy as np

gatherer = GatherStatistics(rank_range=(1, 12), sample_size=5, step=1)

cr, t = gatherer.run()

np.savez("data.npz", cr, t)

plt.plot(cr, t)
plt.xlabel("Number of ranks")
plt.ylabel("Time to solve PDE")
plt.title("Time to solve PDE for different number of ranks")
plt.show()

