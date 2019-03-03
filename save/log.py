# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np

f = open("log.txt", "r")

losses, acces = [], []
for line in f.readlines():
	line = line.strip().split(",")
	if len(line) != 3:
		continue
	loss = float(line[1].split(":")[1])
	losses.append(loss)
	acc = float(line[2].split(":")[1][:-1])
	acces.append(acc)

# plt.figure(figsize=(8,6))
# plt.plot(np.arange(len(losses)), losses)
# plt.savefig("../img/loss.jpg")

plt.figure(figsize=(8,6))
plt.plot(np.arange(len(acces)), acces)
plt.savefig("../img/acc.jpg")






















