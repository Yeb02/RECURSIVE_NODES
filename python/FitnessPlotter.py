from FitnessData import data

import matplotlib.pyplot as plt

x = range(len(data[0]))

for i in range(len(data)):
    plt.plot(x, data[i], "-", label=f"{i}")
    print(sum(data[i]))

plt.legend()
plt.show()