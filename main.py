import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DELAY = 1.1940002e-05

min_max_rat = {}

def get_extremal(data):
    mean_value = data.mean()
    # Calculate distances from the mean
    distances = abs(data - mean_value)

    extremal_index = distances.idxmax()
    extremal_point = data[extremal_index]
    return extremal_point


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

    # Plot columns 4 and 5
    time = df.iloc[:, 3]
    voltage = df.iloc[:, 4]
    voltage_around_delay = voltage[(time < DELAY*1.2) & (time > DELAY*0.8)]
    min_voltage = min(voltage)
    extremal_voltage = get_extremal(voltage_around_delay)
    print(extremal_voltage/min_voltage)
    if should_plot:
        plt.plot(time, voltage)
        plt.xlabel("Time")
        plt.ylabel("Voltage")
        plt.title("Voltage by Time, Around Signal Start")
        plt.legend()
        plt.show()
    return extremal_voltage / min_voltage

if __name__ == "__main__":
    dir = r"C:\Physics\Year 2\Lab\Delay Lines\Week 1"
    for resistance in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
    # for filename in ["0 ohm", "100 ohm", "200 ohm", "300 ohm", "400 ohm",
    #                  "500 ohm", "600 ohm", "700 ohm", "800 ohm", "900 ohm"]:
        rat = plot_scope(filename=fr"{dir}\{resistance} ohm new.csv", should_plot=False)
        min_max_rat[resistance] = rat

    z = get_characteristic_impedance(min_max_rat)
    plt.plot(min_max_rat.keys(), min_max_rat.values())
    fitted_voltages = [return_factor_model(r, z) for r in min_max_rat.keys()]
    plt.plot(min_max_rat.keys(), fitted_voltages, label=f"Characteristic Impedance: {z:.2f}")
    plt.title("Return by resistance")
    plt.xlabel("Resistance [ohm]")
    plt.ylabel("Return factor")
    plt.legend()
    plt.show()

