import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

plt.style.use("physrev.mplstyle")


DELAY_BETWEEN_STEPS = 1.83e-7  # sec
step_peaks = {}
time_until_return = {}


def exponential_model(x, A, gamma):
    return A * np.e ** (gamma * x)

def find_n_peaks(df, step, n, dist):
    # Detect local maxima manually
    time_per_index = df["time"][1] - df["time"][0]
    selected_peaks = []
    remaining_df = df.copy()
    to_start = step * 2
    to_end = (32 - step) * 2

    for _ in range(n):
        # Find the index of the maximum value in the remaining data
        max_idx = remaining_df['voltage'].idxmax()

        # Append the peak index to the list of selected peaks
        selected_peaks.append(max_idx)

        approx_next_peak = to_start if _ % 2 else to_end
        if step == 0:
            approx_next_peak = 32

        # Remove data up to the next zero
        remaining_df = df.iloc[(max_idx + round(0.6*approx_next_peak*DELAY_BETWEEN_STEPS/time_per_index)):]
        # remaining_df = df.iloc[next_zero_idx:]
        # remaining_df = df.iloc[(max_idx + dist):]

        # Break if there are no more peaks to find
        if remaining_df.empty:
            break

    return sorted(selected_peaks)


def plot_scope(filename, should_plot=True):
    df = pd.read_csv(filename)
    df = df.iloc[:, [9, 10]]
    df.columns = ["time", "voltage"]
    step = int(filename.split("\\")[-1].split(" ")[0])

    peak_indexes = find_n_peaks(df, step, 4, 100)  # Adjust `distance` and `height` as needed
    step_peaks[step] = [(df["time"][ind], df["voltage"][ind]) for ind in peak_indexes]
    if should_plot:
        plt.plot(df["time"], df["voltage"],
                 # alpha=0.7,
                 label=f"Start Step = {step}")
        plt.scatter(df["time"][peak_indexes], df["voltage"][peak_indexes], s=10)
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        plt.title(f"Voltage by Time, Infinite Returns")
                  # f".\n Steps = {step}")
        plt.legend()
        # plt.show()


def plot_decay():
    step_peaks_2 = {}
    for step, peaks in step_peaks.items():
        for i, p in enumerate(peaks):
            time, voltage = p
            steps_to_start = min(step, 32-step)*2
            steps_to_end = max(step, 32-step)*2
            steps_passed = step + sum([steps_to_end if j%2 == 0 else steps_to_start for j in range(1,i+1)])
            if step == 0:
                steps_passed = 32 * i
            step_peaks_2.setdefault(steps_passed, []).append(float(voltage))
    x = np.array([])
    y = np.array([])
    for k, values in step_peaks_2.items():
        for v in values:
            x = np.append(x, k)
            y = np.append(y, v)

    normalization_factor = 1 / y[np.where(x == 0)[0]][0]
    normalized_y = [_*normalization_factor for _ in y]
    plt.scatter(x, normalized_y, s=10)
    params, _ = curve_fit(exponential_model, x, normalized_y, p0=(2, -0.02))
    A, gamma = params
    print(A, gamma)
    fit_x = np.linspace(min(x) * 0.9, max(x) * 1.1, 100)
    plt.plot(fit_x, exponential_model(fit_x, A, gamma),
             label=f"Exponential Fit: $A = {round(A*1e4)/1e4}e^{{{round(gamma*1e4)/1e4}s}}$",
             color="orange",
             linestyle="--")
    # plt.title("Voltage by Step, Infinite Returns")
    plt.xlabel("Step")
    plt.ylabel("Normalized Amplitude [V]")
    plt.legend()
    plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Graphs\Decay Between Steps - Infinite Returns.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    dir_name = r"C:\Physics\Year 2\Lab\Delay Lines\Week 2\Infinite Returns 2"
    # for step in [2, 8, 14, 20]:
    for step in [0,
                 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                 # 22, 24, 26, 28, 30
                 ]:
        plot_scope(filename=fr"{dir_name}\{step} steps.csv", should_plot=True)
    # plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Graphs\Infinite Returns Scope.png", dpi=300)
    plt.show()
    plot_decay()
    # plot_return_factor_by_resistance(return_factor_by_resistance)
    # plot_time_until_return(time_until_return)
