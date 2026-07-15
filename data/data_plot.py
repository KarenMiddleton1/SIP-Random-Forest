import json
import matplotlib.pyplot as plt
import numpy as np

with open("formatted_hourly_PVGIS.json") as f:
    data = json.load(f)

gbI = []
gbI_june22 = []
gbI_june23 = []

for i in range(len(data["outputs"]["hourly"])):
    gbI.append(data["outputs"]["hourly"][i]["Gb(i)"])
    if "202206" in data["outputs"]["hourly"][i]["time"]:
        gbI_june22.append(data["outputs"]["hourly"][i]["Gb(i)"])
    if "202306" in data["outputs"]["hourly"][i]["time"]:
        gbI_june23.append(data["outputs"]["hourly"][i]["Gb(i)"])


plt.plot(gbI_june22,color="orange")
plt.title("Global Irradiance (Gb(i)) over Time")
plt.xlabel("Hour")
plt.ylabel("Global Irradiance (W/m^2)")
plt.grid()
plt.savefig("gi_june22.png")

plt.clf()  # Clear the current figure before plotting the next one

plt.plot(gbI_june23,color="blue")
plt.title("Global Irradiance (Gb(i)) over Time")
plt.xlabel("Hour")
plt.ylabel("Global Irradiance (W/m^2)")
plt.grid()
plt.savefig("gi_june23.png")