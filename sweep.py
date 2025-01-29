## THIS ALLOWS US TO FIND CUTOFF FREQUENCY

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

plt.style.use("physrev.mplstyle")


voltage_by_freq = {}
offset_by_freq = {}

sweep_times = {
    "200hz-100khz": 0.12,
    "80khz-600khz": 0.016
}

maxima = {
    4.488 * 1e4: [4.440 * 1e4, 4.580 * 1e4],
    1.4067 * 1e5: [1.3961 * 1e5, 1.4228 * 1e5],
    2.3254 * 1e5: [2.2881 * 1e5, 2.3425 * 1e5],
    3.2636 * 1e5: [3.2325 * 1e5, 3.2770 * 1e5],
    4.081 * 1e5: [4.006 * 1e5, 4.129 * 1e5],
    5.021 * 1e5: [4.952 * 1e5, 5.083 * 1e5],
    5.8 * 1e5: [5.757 * 1e5, 5.832 * 1e5]
}


def linear_model(x, m, b):
    return m*x + b

def get_freq(freq_str):
    if "mhz" in freq_str:
        return float(freq_str.split("mhz")[0]) * 1e6
    elif "khz" in freq_str:
        return float(freq_str.split("khz")[0]) * 1000
    elif "hz" in freq_str:
        return float(freq_str.split("hz")[0])
    raise "Couldn't Parse"


def add_frequency_column(df, start_freq, end_freq, sweep_duration):
    # Keep only rows where time >= 0
    df = df[df["time"] >= 0].reset_index(drop=True)
    df = df[df["time"] <= sweep_duration]

    # Calculate the frequency slope (linear growth)
    freq_slope = (end_freq - start_freq) / sweep_duration

    # Compute the frequency for each time point
    df["frequency"] = start_freq + freq_slope * df["time"]

    return df


def plot_scope(filename, should_plot=True):
    df = pd.read_csv(filename)
    df = df.iloc[:, [9, 10]]
    df.columns = ["time", "voltage"]
    df2 = pd.read_csv(filename)
    df2 = df2.iloc[:, [3, 4]]
    df2.columns = ["time", "voltage"]
    freq_range = (filename.split("\\")[-1].split(".csv")[0].split(" ")[0])
    start_freq = float((re.match(r'^\d+', freq_range.split("-")[0]).group(0)))
    end_freq = float((re.match(r'^\d+', freq_range.split("-")[1]).group(0)))
    time_per_index = df["time"][1] - df["time"][0]
    # period = 1 / (1000 * freq)
    zero_index = df["time"].abs().idxmin()
    # voltage_by_freq[freq] = max(df["voltage"])
    # offset_by_freq[freq] = df["time"][df.iloc[zero_index:zero_index+round(period/time_per_index)]["voltage"].idxmax()] - df["time"][zero_index]
    if should_plot:
        plt.plot(df["time"], df["voltage"])
        # plt.plot(df2["time"], df2["voltage"])
        # plt.scatter(df["time"][peak_indexes], df["voltage"][peak_indexes])
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        # plt.title(f"Voltage by Time, Different Frequencies.\n Frequency = {freq}")
        plt.legend()
        plt.show()


def plot_scope_with_frequency(filename, should_plot=True):
    df = pd.read_csv(filename)
    df = df.iloc[:, [9, 10]]
    df.columns = ["time", "voltage"]
    df2 = pd.read_csv(filename)
    df2 = df2.iloc[:, [3, 4]]
    df2.columns = ["time", "voltage"]

    # Extract frequency range from filename
    freq_range = (filename.split("\\")[-1].split(".csv")[0].split(" ")[0])
    sweep_duration = sweep_times[freq_range]
    freqs = freq_range.split("-")
    start_freq = get_freq(freqs[0])
    end_freq = get_freq(freqs[1])

    df = add_frequency_column(df, start_freq, end_freq, sweep_duration)

    if should_plot:
        # Plotting using frequency axis
        plt.plot(df["frequency"], df["voltage"], label="Circuit Center")
        # plt.plot(df2["frequency"], df2["voltage"], label="Dataset 2")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Voltage [V]")
        plt.title("Voltage By Frequency, Sweep")
        plt.legend()
        plt.savefig(fr"C:\Physics\Year 2\Lab\Delay Lines\Graphs\Appendix\Scope by Frequency - {freq_range}.png", dpi=300)
        plt.show()


def plot_voltage_by_freq():
    plt.scatter(voltage_by_freq.keys(), voltage_by_freq.values())
    plt.xlabel("Frequency [KHz]")
    plt.ylabel("Amplitude [V]")
    plt.title("Amplitude by Frequency, Sine Wave")
    plt.legend()
    # plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Week 2\Amplitude By Frequency.png", dpi=300)

    plt.show()


def plot_offset_by_freq():
    plt.scatter(offset_by_freq.keys(), offset_by_freq.values())
    plt.xlabel("Frequency [KHz]")
    plt.ylabel("Time Offset [sec]")
    plt.title("Time Offset by Frequency, Sine Wave")
    plt.legend()
    # plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Week 3\Time Offset By Frequency.png", dpi=300)
    plt.show()


def plot_maxima():
    x = [i+1 for i in range(len(maxima))]
    y = np.array(list(maxima.keys()))
    y_errors = [
        [(y[ind] - e[0]) *1e-3 for ind, e in enumerate(list(maxima.values()))],
        [(e[1] - y[ind]) *1e-3 for ind, e in enumerate(list(maxima.values()))]
    ]

    popt, pcov = curve_fit(linear_model, x, y)

    # Get the fitted line
    fitted_y = linear_model(np.array(x), *popt)
    fitted_y *= 1e-3
    y *= 1e-3
    plt.errorbar(x, y, yerr=y_errors, fmt='o', markersize=4)
    plt.plot(x, fitted_y, label=f"Linear Fit: $f = {popt[0]*1e-3:.2f}n - {-1*popt[1]*1e-3:.2f}$")
    plt.xlabel("Peak Number")
    plt.ylabel("Source Frequency [kHz]")
    # plt.title("Frequencies of Peak Numbers")
    plt.legend()
    plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Graphs\Amplitude Maxima.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    dir_name = r"C:\Physics\Year 2\Lab\Delay Lines\Week 3\Sweeps"
    for sweep_range in ["200hz-100khz", "80khz-600khz",]:
        # plot_scope(filename=fr"{dir_name}\{sweep_range}.csv", should_plot=True)
        plot_scope_with_frequency(filename=fr"{dir_name}\{sweep_range}.csv", should_plot=True)
    plot_maxima()
    # plot_voltage_by_freq()
    # plot_offset_by_freq()
