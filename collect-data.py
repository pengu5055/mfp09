from gatherstat import GatherStatistics
import matplotlib.pyplot as plt
import numpy as np

# gatherer = GatherStatistics(rank_range=(1, 13), sample_size=10, step=1)

# cr, t = gatherer.run()

# np.savez("data_full-range_manual.npz", cr, t)

with np.load("data_full-range.npz") as data:
    cr = data["arr_0"]
    t = data["arr_1"]

plt.rcParams.update({'font.family': 'Verdana'})
fig, ax = plt.subplots(facecolor="#4d4c4c")
plt.plot(cr, t, c="#a8325e")
plt.xlabel("Number of ranks")
plt.ylabel("Time to solve PDE")
plt.suptitle("ColocationSolver.solve_Properly() MPI Scaling", color="#dedede")
ax.set_facecolor("#bababa")
plt.grid(c="#d1d1d1", alpha=0.5)
ax.spines['bottom'].set_color("#dedede")
ax.spines['top'].set_color("#dedede")
ax.spines['right'].set_color("#dedede")
ax.spines['left'].set_color("#dedede")
ax.xaxis.label.set_color("#dedede")
ax.yaxis.label.set_color("#dedede")
ax.tick_params(axis="x", colors="#dedede")
ax.tick_params(axis="y", colors="#dedede")
#ax.axhline(0, linestyle="--", color="#dedede")
#plt.subplots_adjust(right=1)
plt.show()

