import pandas as pd
import matplotlib.pyplot as plt

DELAY = 1.1940002e-05

min_max_rat = {}

def get_extremal(data):
    mean_value = data.mean()
    # Calculate distances from the mean
    distances = abs(data - mean_value)

    extremal_index = distances.idxmax()
    extremal_point = data[extremal_index]
    return extremal_point

def plot_scope(filename):
    df = pd.read_csv(filename)

    # Plot columns 4 and 5
    time = df.iloc[:, 3]
    voltage = df.iloc[:, 4]
    voltage_around_delay = voltage[(time < DELAY*1.2) & (time > DELAY*0.8)]
    min_voltage = min(voltage)
    extremal_voltage = get_extremal(voltage_around_delay)
    print(extremal_voltage/min_voltage)
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
        rat = plot_scope(filename=fr"{dir}\{resistance} ohm new.csv")
        min_max_rat[resistance] = rat

    plt.plot(min_max_rat.keys(), min_max_rat.values())
    plt.title("Ratio")
    plt.show()

