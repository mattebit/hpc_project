import re
import matplotlib.pyplot as plt
import numpy as np


FILENAME = "log.log"

f = open(FILENAME, "r")

pattern = "\[ID:(\d+)\]\sTime\staken:\s(\d+\.\d+)\ssec\s\((.*)\)"

values = {}

for l in f:
    r = re.compile(pattern)
    res = r.match(l)
    if res is not None:
        id = res.group(1)
        time = res.group(2)
        desc = res.group(3)
        if values.get(desc) is None:
            values[desc] = [float(time)]
        else:
            values[desc].append(float(time))
        print(f"id: {id}, time: {time}, desc:{desc}")

maxx = 0
for i in values:
    if len(values[i]) > maxx:
        maxx = len(values[i])

x = [i for i in range(1,maxx)]

plt.plot(x, values["2 All gather"], label="All gather 2")
#plt.plot(x, values["Find root"])
plt.plot(x, values["Prune graph"], label="Prune graph")
plt.xlabel("Cycle")
plt.ylabel("Time in seconds")
plt.legend()
plt.show()