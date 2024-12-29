## THIS ALLOWS US TO FIND CUTOFF FREQUENCY

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

plt.style.use("ggplot")


voltage_by_freq = {}
offset_by_freq = {}


def plot_scope(filename, should_plot=True):
    df = pd.read_csv(filename)
    df = df.iloc[:, [3, 4]]
    df.columns = ["time", "voltage"]
    freq = float(filename.split("\\")[-1].split(".csv")[0])
    time_per_index = df["time"][1] - df["time"][0]
    period = 1 / (1000 * freq)
    zero_index = df["time"].abs().idxmin()
    voltage_by_freq[freq] = max(df["voltage"])
    offset_by_freq[freq] = df["time"][df.iloc[zero_index:zero_index+round(period/time_per_index)]["voltage"].idxmax()] - df["time"][zero_index]
    if should_plot:
        plt.plot(df["time"], df["voltage"])
        # plt.scatter(df["time"][peak_indexes], df["voltage"][peak_indexes])
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.title(f"Voltage by Time, Different Frequencies.\n Frequency = {freq}")
        plt.legend()
        plt.show()


def plot_voltage_by_freq():
    plt.scatter(voltage_by_freq.keys(), voltage_by_freq.values())
    plt.xlabel("Frequency [KHz]")
    plt.ylabel("Amplitude [V]")
    plt.title("Amplitude by Frequency, Sine Wave")
    plt.legend()
    plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Week 2\Amplitude By Frequency.png", dpi=300)

    plt.show()


def plot_offset_by_freq():
    plt.scatter(offset_by_freq.keys(), offset_by_freq.values())
    plt.xlabel("Frequency [KHz]")
    plt.ylabel("Time Offset [sec]")
    plt.title("Time Offset by Frequency, Sine Wave")
    plt.legend()
    plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Week 2\Time Offset By Frequency.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    dir_name = r"C:\Physics\Year 2\Lab\Delay Lines\Week 2\Sine"
    for frequency in [40, 41, 42, 43, 44, 44.5, 45, 45.5, 46, 47, 48, 49, 50]:
        plot_scope(filename=fr"{dir_name}\{frequency}.csv", should_plot=False)
    plot_voltage_by_freq()
    plot_offset_by_freq()
