import pandas as pd
import matplotlib.pyplot as plt


FILENAME = "HPC_stats - Foglio1.csv"

df = pd.read_csv(FILENAME)

# Print the first few rows
#print(df.head())

# Print information about the DataFrame
#print(df.info())

df["line1"] = 0.75

df20 = df[df["# Nodes"] == 20000]
df20 = df20.groupby(["OpenMP process"]).agg({
    "MPI process": "mean",
    "Exec Time Parallel (s)": "mean",
    "Exec Time serial (s)": "mean",
    "Speedup": "mean",
    "Efficiency (MPI)": "mean",
    "Efficieny (MPI+OpenMP)": "mean"
})
print(df20)
df40 = df[df["# Nodes"] == 40000]
df40 = df40.groupby(["OpenMP process"]).agg({
    "MPI process": "mean",
    "Exec Time Parallel (s)": "mean",
    "Exec Time serial (s)": "mean",
    "Speedup": "mean",
    "Efficiency (MPI)": "mean",
    "Efficieny (MPI+OpenMP)": "mean"
})
df80 = df[df["# Nodes"] == 80000]
df80 = df80.groupby(["OpenMP process"]).agg({
    "MPI process": "mean",
    "Exec Time Parallel (s)": "mean",
    "Exec Time serial (s)": "mean",
    "Speedup": "mean",
    "Efficiency (MPI)": "mean",
    "Efficieny (MPI+OpenMP)": "mean"
})

#print(df20["Exec Time Parallel (s)"])

# Line plot
plt.plot(df20['MPI process'], df20['Speedup'], marker="o", label='20k')
plt.plot(df40['MPI process'], df40['Speedup'], marker="o", label='40k')
plt.plot(df80['MPI process'], df80['Speedup'], marker="o", label='80k')
plt.xlabel('MPI processes')
plt.ylabel('Speedup')
plt.legend()
plt.show()

plt.clf()

plt.plot(df20['MPI process'], df20['Efficiency (MPI)'], marker="o", label='20k')
plt.plot(df40['MPI process'], df40['Efficiency (MPI)'], marker="o", label='40k')
plt.plot(df80['MPI process'], df80['Efficiency (MPI)'], marker="o", label='80k')
#plt.plot(df80['MPI process'], df80['line1'], linestyle='--')
plt.xlabel('MPI processes')
plt.ylabel('Efficiency MPI')
plt.legend()
plt.show()

plt.clf()

plt.plot(df20['MPI process'], df20['Efficieny (MPI+OpenMP)'], marker="o", label='20k')
plt.plot(df40['MPI process'], df40['Efficieny (MPI+OpenMP)'], marker="o", label='40k')
plt.plot(df80['MPI process'], df80['Efficieny (MPI+OpenMP)'], marker="o", label='80k')
plt.xlabel('MPI processes')
plt.ylabel('Efficiency MPI+openMP')
plt.legend()
plt.show()