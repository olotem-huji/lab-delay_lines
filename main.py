import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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


def return_factor_model(R, Z_0):
    return (R - Z_0)/(R + Z_0)


def get_characteristic_impedance(return_by_resistance):
    initial_guess = [330]  # Initial guess for Z_0
    params, covariance = curve_fit(return_factor_model, np.array(list(return_by_resistance.keys())),
                                   np.array(list(return_by_resistance.values())), p0=initial_guess)

    # Extract the fitted parameter
    Z_0_fit = params[0]
    print(f"Fitted Z_0: {Z_0_fit}")
    return Z_0_fit


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
        plt.title(f"Voltage by Time, Around Signal Start.\n Resistance = {resistance} ohm")
        plt.legend()
        plt.show()


def plot_return_factor_by_resistance(data):
    z = get_characteristic_impedance(data)
    plt.plot(data.keys(), data.values())
    fitted_voltages = [return_factor_model(r, z) for r in data.keys()]
    plt.plot(data.keys(), fitted_voltages, label=f"Characteristic Impedance: {z:.2f} ohm")
    plt.title("Return Factor by Resistance")
    plt.xlabel("Resistance [ohm]")
    plt.ylabel("Return Factor [%]")
    plt.legend()
    plt.show()


def plot_time_until_return(data):
    plt.plot(data.keys(), data.values())
    plt.title("Time Until Return By Resistance")
    plt.xlabel("Resistance [ohm]")
    plt.ylabel("Time Until Return [s]")
    plt.show()


if __name__ == "__main__":
    dir_name = r"C:\Physics\Year 2\Lab\Delay Lines\Week 1"
    for resistance in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
        plot_scope(filename=fr"{dir_name}\{resistance} ohm new.csv", should_plot=False)
    plot_return_factor_by_resistance(return_factor_by_resistance)
    plot_time_until_return(time_until_return)
