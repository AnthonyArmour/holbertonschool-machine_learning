#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Generate our data
x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

# Label the axes and title
plt.xlabel("Time (years)")
plt.ylabel("Fractions Remaining")
plt.suptitle("Exponential Decay of Radioactive Elements")

# Plot figure
plt.xlim(0, 20000)
plt.plot(x, y2, "g-", label="Ra-226")
plt.plot(x, y1, "r--", label="C-14")

# Show or save plot
plt.show()
# plt.savefig("filename.jpg")
