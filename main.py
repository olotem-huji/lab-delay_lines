import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.style.use("physrev.mplstyle")
DELAY = 1.1940002e-05

return_factor_by_resistance = {}
time_until_return = {}

def get_extremal(data):
    mean_value = data.mean()
    # Calculate distances from the mean
    distances = abs(data - mean_value)

    extremal_index = distances.idxmax()
    extremal_point = data[extremal_index]
    return extremal_point, extremal_index


def return_factor_model(R, Z_0, a):
    return (R - Z_0)/(R + Z_0) * a


def get_characteristic_impedance(return_by_resistance):
    initial_guess = [440, 0.4]  # Initial guess for Z_0
    params, covariance = curve_fit(return_factor_model, np.array(list(return_by_resistance.keys())),
                                   np.array(list(return_by_resistance.values())), p0=initial_guess)

    # Extract the fitted parameter
    Z_0_fit = params[0]
    a_fit = params[1]
    print(f"Fitted Z_0: {Z_0_fit}")
    return Z_0_fit, a_fit


def plot_scope(filename, should_plot=True):
    df = pd.read_csv(filename)
    resistance = int(filename.split("\\")[-1].split(" ")[0])

    time = df.iloc[:, 3]
    voltage = df.iloc[:, 4]
    voltage_around_delay = voltage[(time < DELAY*1.2) & (time > DELAY*0.8)]
    min_voltage = min(voltage)
    extremal_voltage, extremal_idx = get_extremal(voltage_around_delay)
    if resistance == 400:
        extremal_voltage, extremal_idx = (max(voltage_around_delay), voltage_around_delay.idxmax())
    return_factor_by_resistance[resistance] = extremal_voltage / min_voltage
    time_until_return[resistance] = time[extremal_idx] - time[voltage.idxmin()]

    print(time[extremal_idx])
    if should_plot:
        plt.plot(time, voltage)
        plt.xlabel("Time [s]")
        plt.ylabel("Voltage [V]")
        # plt.title(f"Voltage by Time, Around Signal Start.\n Resistance = {resistance} ohm")
        plt.legend()
        if "100" in filename:
            plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Graphs\Appendix\Amplitude By Resistance 100 ohm.png",
                        dpi=300)
        if "400" in filename:
            plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Graphs\Appendix\Amplitude By Resistance 400 ohm.png",
                        dpi=300)
        plt.show()


def plot_return_factor_by_resistance(data):
    z, a = get_characteristic_impedance(data)
    plt.scatter(data.keys(), data.values(), s=10)
    fitted_x = np.linspace(min(data.keys())*0.95, max(data.keys())*1.05, 100)
    fitted_voltages = return_factor_model(fitted_x, z, a)
    plt.plot(fitted_x, fitted_voltages,
             label=(fr"Rational Fit: $\Gamma = {a:.2f}\frac{{R - Z_0}}{{R + Z_0}}$" + "\n" + fr"$Z_0$ = {z:.2f} $\Omega$"),
             linestyle="--",
             color="orange")
    # plt.title("Return Factor by Resistance")
    plt.xlabel(r"Resistance [$\Omega$]")
    plt.ylabel("Return Factor [%]")
    plt.legend()
    plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Graphs\Return Factor By Resistance.png", dpi=300)
    plt.show()


def plot_time_until_return(data):
    plt.scatter(data.keys(), data.values(), s=10)
    # plt.title("Time Until Return By Resistance")
    plt.xlabel(r"Resistance [$\Omega$]")
    plt.ylabel(r"Time Until Return [$s$]")
    plt.savefig(r"C:\Physics\Year 2\Lab\Delay Lines\Graphs\Appendix\Time Until Return.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    dir_name = r"C:\Physics\Year 2\Lab\Delay Lines\Week 1"
    for resistance in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
        plot_scope(filename=fr"{dir_name}\{resistance} ohm new.csv", should_plot=False)
    plot_return_factor_by_resistance(return_factor_by_resistance)
    plot_time_until_return(time_until_return)
