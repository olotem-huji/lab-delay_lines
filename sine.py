## THIS ALLOWS US TO FIND CUTOFF FREQUENCY

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.style.use("physrev.mplstyle")


voltage_by_freq = {}
circuit_voltage_by_freq = {}
voltage_by_freq_by_step = {}
offset_by_freq = {}


def square_model(x, a, b, c):
    # return ((a*x + b)**2)
    return a*x**2 + b*x + c

def lorentzian_pdf(x, A, x0, gamma):
    return A * (gamma) / ((x - x0)**2 + gamma**2)

def plot_scope(filename, should_plot=True):
    df = pd.read_csv(filename)
    df = df.iloc[:, [9, 10]]
    df.columns = ["time", "voltage"]
    df2 = pd.read_csv(filename)
    df2 = df2.iloc[:, [3, 4]]
    df2.columns = ["time", "voltage"]
    freq = float(filename.split("\\")[-1].split(".csv")[0].split(" ")[0])
    step = int(filename.split("\\")[-1].split(".csv")[0].split(" ")[-1])
    time_per_index = df["time"][1] - df["time"][0]
    period = 1 / (1000 * freq)
    zero_index = df["time"].abs().idxmin()
    voltage_by_freq[freq] = max(df["voltage"])
    circuit_voltage_by_freq[freq] = max(df2["voltage"])
    voltage_by_freq_by_step.setdefault(freq, {})[step] = max(df["voltage"])
    offset_by_freq[freq] = df["time"][df.iloc[zero_index:zero_index+round(period/time_per_index)]["voltage"].idxmax()] - df["time"][zero_index]
    if should_plot:
        plt.plot(df["time"], df["voltage"])
        plt.plot(df2["time"], df2["voltage"])
        # plt.scatter(df["time"][peak_indexes], df["voltage"][peak_indexes])
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.title(f"Voltage by Time, Different Frequencies.\n Frequency = {freq}")
        plt.legend()
        plt.show()


def plot_voltage_by_freq():
    x = list(voltage_by_freq.keys())
    y = list(voltage_by_freq.values())
    plt.scatter(x, y, s=10)
    # plt.scatter(circuit_voltage_by_freq.keys(), circuit_voltage_by_freq.values(),
    #             label="Circuit voltage")

    popt, pcov = curve_fit(lorentzian_pdf, x, y, p0=[max(y), 44.5, 1])
    fitted_x = np.linspace(min(x)*0.95, max(x)*1.05, 100)
    fitted_y = lorentzian_pdf(fitted_x, *popt)
    plt.plot(fitted_x, fitted_y,
             label=fr"Lorentzian Fit: $A = C*\frac{{\gamma}}{{(f-f_0)^2 + \gamma^2}}$" +
                   "\n" +
                   fr"$C={popt[0]:.2f}$, $f_0={popt[1]:.2f}$, $\gamma={popt[2]:.2f}$",
             color="orange",
             linestyle="--"
             # label=f"Square Fit: y = ({popt[0]:.2f}x + {popt[1]:.2f})**2"
             )

    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Amplitude [V]")
    # plt.title("Amplitude by Frequency, Sine Wave")
    plt.legend()
    plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Graphs\Amplitude By Frequency Lorentzian.png", dpi=300)

    plt.show()

def plot_voltage_by_freq_by_step():
    for freq in voltage_by_freq_by_step.keys():
        plt.scatter(
            voltage_by_freq_by_step[freq].keys(),
            voltage_by_freq_by_step[freq].values(),
            label=f"Frequency = {freq}"
        )
    plt.xlabel("Step")
    plt.ylabel("Amplitude [V]")
    plt.legend()
    plt.show()

def plot_offset_by_freq():
    plt.scatter(offset_by_freq.keys(), offset_by_freq.values(), s=10)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Time Offset [sec]")
    plt.title("Time Offset by Frequency, Sine Wave")
    plt.legend()
    plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Graphs\Time Offset By Frequency.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    dir_name = r"C:\Physics\Year 2\Lab\Delay Lines\Week 3\Sine"
    for frequency in [40, 41, 42, 43, 44, 44.5, 45, 46, 47, 48, 49, 50]:
        plot_scope(filename=fr"{dir_name}\{frequency} khz step 15.csv", should_plot=False)
    plot_voltage_by_freq()
    # for step in [15, 17, 19]:
    #     plot_scope(filename=fr"{dir_name}\45 khz step {step}.csv")
    for frequency in [43, 44, 44.5, 45, 46, 47]:
        for step in [17, 19, 15]:
            plot_scope(filename=fr"{dir_name}\{frequency} khz step {step}.csv", should_plot=False)
    plot_voltage_by_freq_by_step()
    plot_offset_by_freq()
