#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
xtick = np.arange(0, 101, 10)
ytick = np.arange(0, 31, 5)
student_grades = np.random.normal(68, 15, 50)

plt.suptitle("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")

plt.xlim(0, 100)
plt.ylim(0, 30)
plt.xticks(xtick)
plt.yticks(ytick)
plt.hist(student_grades, range=(0, 100))
plt.show()
