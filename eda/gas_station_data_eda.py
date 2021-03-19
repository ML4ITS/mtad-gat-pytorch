import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import data
df = pd.read_csv("../datasets/gas_sensor_data.csv", delimiter=",")
ds = df.drop(["Time"], axis=1)

# Visualize trends in data
sns.set_style("darkgrid")
ds.plot(kind="line", legend="reverse", title="Visualizing Sensor Array Time-Series")
plt.legend(loc="upper right", shadow=True, bbox_to_anchor=(1.2, 0.8))
plt.show()

# Dropping Temperature & Relative Humidity as they do not change with Time
ds.drop(["Temperature", "Rel_Humidity"], axis=1, inplace=True)

# Again Visualizing the time-series data
sns.set_style("darkgrid")
ds.plot(kind="line", legend="reverse", title="Visualizing Sensor Array Time-Series")
plt.legend(loc="upper right", shadow=True, bbox_to_anchor=(1.1, 0.8))
plt.show()
