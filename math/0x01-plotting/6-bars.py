#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

x_lables = ['Farrah', "Fred", "Felicia"]
apples = fruit[0, :]
bananas = fruit[1, :]
oranges = fruit[2, :]
peaches = fruit[3, :]

plt.bar(x_lables, apples, color='r', width=0.5)
plt.bar(x_lables, bananas, bottom=apples, color="yellow", width=0.5)
plt.bar(x_lables, oranges, bottom=apples+bananas, color="orange", width=0.5)
plt.bar(x_lables, peaches, bottom=apples+bananas+oranges,
        color="wheat", width=0.5)

plt.ylabel("Quantity of Fruit")
plt.suptitle("Number of Fruit per Person")
plt.xticks(x_lables)
plt.ylim(0, 80, 80)
plt.legend(["apples", "bananas", "oranges", "peaches"])
plt.show()
