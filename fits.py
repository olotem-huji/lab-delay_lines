from scipy.optimize import curve_fit
import numpy as np
from enum import Enum

def linear_function(x, a, b):
    return np.add(np.multiply(x, a), b)

def inverse_decay(x, y0):
    return y0/x
def inverse_square_decay(x, y0):
    return y0 / np.power(x, 2)

def inverse_power_decay(x, a, y0):
    return y0 / np.power(x, a)

def exponential_decay(x, a, tau, b):
    return a * np.exp(np.multiply(x, -1 * tau)) + b
def combined_decay(x, y0, tau):
    return y0 / x**2 * np.exp(np.multiply(x,-1* tau))

def inverse_exponential_decay(x, y0, tau):
    return y0/x * np.exp(np.multiply(x, -1* tau))

def inverse_power_exponential_decay(x, y0, a, tau):
    return y0/(x**a) * np.exp(np.multiply(x, -1 * tau))


def calculate_r_squared(x, y, popt, f):
    residuals = y - f(x, *popt)
    mean_y = np.mean(y)
    sst = np.sum((y - mean_y) ** 2)
    ssr = np.sum(residuals ** 2)
    r_squared = 1 - (ssr/sst)
    return r_squared

def fit_decay(x,y,method):
    np_x = np.array(x)
    np_y = np.array(y)
    if method == FitTypes.LINEAR:
        popt, pcov = curve_fit(linear_function, np.multiply(np_x, 1), np_y)
        a, b = popt
        x_fit = np.linspace(min(np_x), max(np_x), 100)
        y_fit = linear_function(x_fit, a, b)  # coefficients = np.polyfit(np_x, np_y, 2)
        eq = f"{round(a * 100) / 100}*x + {round(b*100)/100}"
        r_squared = calculate_r_squared(x,y, popt, linear_function)
    elif method == FitTypes.INVERSE:
        popt, pcov = curve_fit(inverse_decay, np.multiply(np_x, 1), np_y)
        y0, = popt
        x_fit = np.linspace(min(np_x), max(np_x), 100)
        y_fit = inverse_decay(x_fit, y0)  # coefficients = np.polyfit(np_x, np_y, 2)
        eq = f"{round(y0 * 100) / 100}/x"
        r_squared = calculate_r_squared(x, y, popt, inverse_decay)
    elif method == FitTypes.INVERSE_SQUARE:
        popt, pcov = curve_fit(inverse_square_decay, np.multiply(np_x, 1), np_y)
        y0, = popt
        x_fit = np.linspace(min(np_x), max(np_x), 100)
        y_fit = inverse_square_decay(x_fit, y0)  # coefficients = np.polyfit(np_x, np_y, 2)
        eq = f"{round(y0 * 100) / 100}/x^2"
        r_squared = calculate_r_squared(x, y, popt, inverse_square_decay)
    elif method == FitTypes.INVERSE_POWER:
        popt, pcov = curve_fit(inverse_power_decay, np.multiply(np_x, 1), np_y)
        a, y0 = popt
        x_fit = np.linspace(min(np_x), max(np_x), 100)
        y_fit = inverse_power_decay(x_fit, a, y0)  # coefficients = np.polyfit(np_x, np_y, 2)
        eq = f"{round(y0 * 100) / 100}/x^{a}"
        r_squared = calculate_r_squared(x, y, popt, inverse_power_decay)
    elif method == FitTypes.EXPONENT:
        popt, pcov = curve_fit(exponential_decay, np.multiply(np_x, 1), np_y)
        a, tau, b = popt
        x_fit = np.linspace(min(np_x), max(np_x), 100)
        y_fit = exponential_decay(x_fit, a, tau, b)  # coefficients = np.polyfit(np_x, np_y, 2)
        eq = f"{round(a * 100) / 100} * e^-{round(tau * 100) / 100}*x + {round(b * 100) / 100}"
        r_squared = calculate_r_squared(x, y, popt, exponential_decay)
    elif method == FitTypes.INVERSE_SQUARE_EXPONENT:
        popt, pcov = curve_fit(combined_decay, np.multiply(np_x, 1), np_y)
        y0, tau = popt
        x_fit = np.linspace(min(np_x), max(np_x), 100)
        y_fit = combined_decay(x_fit, y0, tau)  # coefficients = np.polyfit(np_x, np_y, 2)
        eq = f"{round(y0 * 100) / 100}/x^2 * e^{round(tau * 100) / 100}*x"
        r_squared = calculate_r_squared(x, y, popt, combined_decay)
    elif method == FitTypes.INVERSE_EXPONENT:
        popt, pcov = curve_fit(inverse_exponential_decay, np.multiply(np_x, 1), np_y)
        y0, tau = popt
        x_fit = np.linspace(min(np_x), max(np_x), 100)
        y_fit = combined_decay(x_fit, y0, tau)  # coefficients = np.polyfit(np_x, np_y, 2)
        eq = f"{round(y0 * 100) / 100}/x * e^{round(tau * 100) / 100}*x"
        r_squared = calculate_r_squared(x, y, popt, inverse_exponential_decay)
    elif method == FitTypes.INVERSE_POWER_EXPONENT:
        popt, pcov = curve_fit(inverse_power_exponential_decay, np.multiply(np_x, 1), np_y)
        y0, a, tau = popt
        x_fit = np.linspace(min(np_x), max(np_x), 100)
        y_fit = inverse_power_exponential_decay(x_fit, y0, a, tau)  # coefficients = np.polyfit(np_x, np_y, 2)
        eq = f"{round(y0 * 100) / 100}/(x^{a}) * e^{round(tau * 100) / 100}*x"
        r_squared = calculate_r_squared(x, y, popt, inverse_power_exponential_decay)
    else:
        raise NotImplemented
    return {
        "x_fit": x_fit,
        "y_fit": y_fit,
        "eq": eq,
        "r_squared": r_squared
    }


class FitTypes(Enum):
    LINEAR = "linear"
    INVERSE = "inverse"
    INVERSE_SQUARE = "inverse square"
    INVERSE_POWER = "inverse power"
    EXPONENT = "exponent"
    INVERSE_EXPONENT = "inverse exponent"
    INVERSE_SQUARE_EXPONENT = "combined decay"
    INVERSE_POWER_EXPONENT = "inverse power exponent"
    RETURN = "return"
