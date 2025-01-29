## THIS ALLOWS US TO FIND THE DELAY BETWEEN STEPS


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.style.use("physrev.mplstyle")


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
    y = y - y[0] ## Normalize
    y = y * 1e6
    params, _ = curve_fit(linear_model, x, y)
    slope, intercept = params
    plt.scatter(x, y, s=10)
    fitted_x = np.linspace(min(x)*0.95, max(x)*1.05, 100)
    plt.plot(fitted_x, slope * fitted_x + intercept,
             label=fr"Linear Fit: $\Delta$$t = {slope*10:.2f}*10^{{-7}}s + {intercept*100:.3f}*10^{{-8}}$",
             color="orange",
             linestyle="--")
    plt.xlabel("Step")
    plt.ylabel(r"Peak Offset [$\mu$s]")
    # plt.title("Delay Between Steps")
    plt.legend()
    plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Graphs\Delay Between Steps.png", dpi=300)
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
    plot_peak_offset()
    # plot_peak_decay()
