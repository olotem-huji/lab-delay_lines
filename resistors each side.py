## THIS ALLOWS US TO FIND THE DELAY BETWEEN STEPS


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
print(plt.style.available)
plt.style.use("ggplot")

# plt.rcParams.update({
#     # 'figure.figsize': (6, 4),    # Set figure size (width, height) in inches
#     'figure.dpi': 300,          # Set DPI for high resolution
#     # 'font.size': 12,             # Set font size
#     # 'axes.titlesize': 14,        # Set axes title font size
#     # 'axes.labelsize': 12,        # Set axes labels font size
#     # 'xtick.labelsize': 10,       # Set x-axis tick font size
#     # 'ytick.labelsize': 10,       # Set y-axis tick font size
#     'lines.linewidth': 1,        # Set line width
#     'lines.markersize': 3,       # Set marker size
#     'axes.grid': True,           # Enable grid for better readability
#     'grid.alpha': 0.3,           # Grid transparency
# })


peak_offset = {}
peak_decay = {}


def linear_model(x, a, b):
    return a*x + b


def exponential_model(x, A, gamma):
    return A * np.e ** (gamma * x)


def plot_scope(filename, should_plot=True):
    df = pd.read_csv(filename)
    df = df.iloc[:, [9, 10]]
    df.columns = ["time", "voltage"]
    step = int(filename.split("\\")[-1].split(" ")[0])

    zero_index = df["time"].abs().idxmin()
    peak_offset[step] = df["time"][df["voltage"].idxmax()] - df["time"][zero_index]
    peak_decay[step] = max(df["voltage"])

    if should_plot:
        plt.plot(df["time"], df["voltage"])
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.title(f"Voltage by Time, Infinite Returns.\n Steps = {step}")
        plt.legend()
        plt.show()



def plot_peak_offset():
    x = np.array(list(peak_offset.keys()))
    y = np.array(list(peak_offset.values()))
    params, _ = curve_fit(linear_model, x, y)
    slope, intercept = params
    plt.scatter(x, y)
    plt.plot(x, slope * x + intercept,
             label=f"{round(slope*1e9)/1e9}*x + {round(intercept*1e9)/1e9}",
             color="red", linestyle="--")
    plt.xlabel("Steps")
    plt.ylabel("Peak Offset [sec]")
    plt.title("Delay Between Steps")
    plt.legend()
    plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Week 2\Delay Between Steps.png", dpi=300)
    plt.show()

def plot_peak_decay():
    x = np.array(list(peak_decay.keys()))
    y = np.array(list(peak_decay.values()))
    params, _ = curve_fit(exponential_model, x, y, p0=(0.4, -0.01))
    A, gamma = params
    plt.scatter(x, y)
    # plt.plot(x, exponential_model(x, A, gamma),
    #          label=f"{round(A*1e3)/1e3}*e **({round(gamma*1e5)/1e5} *x)",
    #          color="red", linestyle="--")
    plt.xlabel("Steps")
    plt.ylabel("Peak Voltage [V]")
    plt.title("Decay Between Steps - No Returns")
    plt.legend()
    plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Week 2\Decay Between Steps - No Returns.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    dir_name = r"C:\Physics\Year 2\Lab\Delay Lines\Week 2\Resistors Each Side"
    for step in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]:
        plot_scope(filename=fr"{dir_name}\{step} step diff.csv", should_plot=False)
    # plot_peak_offset()
    plot_peak_decay()
