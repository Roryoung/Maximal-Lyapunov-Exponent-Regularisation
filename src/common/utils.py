"""This module provides a collection of utility functions"""

import os
from typing import Union
import shutil
import json

import numpy as np
from scipy import stats
from torch import nn


class Color:  # pylint: disable=R0903
    """Define optional colors which can be used by a print function"""

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def print_line(string):
    """Print a given string on the current line."""
    term_size = shutil.get_terminal_size()
    print("\r" + " " * term_size.columns, end="", flush=True)
    print(f"\r{string}", end="", flush=True)


def clear_line():
    """Clear the current line"""
    term_size = shutil.get_terminal_size()
    print("\r" + " " * term_size.columns, end="")
    print("\r", end="")


def fancy_print(input_value: Union[str, float]) -> None:
    """Print the given string surrounded by a box"""
    bold = Color.BOLD
    end = Color.END
    tl_corner = "\u250c"
    tr_corner = "\u2510"
    bl_corner = "\u2514"
    br_corner = "\u2518"
    h_dash = "\u2500"
    v_dash = "\u2502"

    string = str(input_value)

    print(f"{bold}{tl_corner}{h_dash * (len(string) + 2)}{tr_corner}{end}")
    print(f"{bold}{v_dash} {string} {v_dash}{end}")
    print(f"{bold}{bl_corner}{h_dash * (len(string) + 2)}{br_corner}{end}")


def underline_print(string: str) -> None:
    """Print the given string underlined"""
    print(f"{Color.UNDERLINE}{string}{Color.END}")


def bold_print(string: str) -> None:
    """Print the given string in bold"""
    print(f"{Color.BOLD}{string}{Color.END}")


def print_hbar() -> None:
    """Print a horizontal bar to the console"""
    term_size = shutil.get_terminal_size()
    print("_" * term_size.columns)
    print()
    print()


def human_format(number: int) -> None:
    """Return a readable version of large numbers"""
    number = float(f"{number:.3g}")
    magnitude = 0
    while abs(number) >= 1000:
        magnitude += 1
        number /= 1000.0
    return f"{number:.1f}".rstrip("0").rstrip(".") + ["", "K", "M", "B", "T"][magnitude]


def machine_format(str_number: str) -> int:
    """Convert human readable numbers to integers"""
    if str_number[-1].isalpha():
        number = float(str_number[:-1])
        magnitude_str = str_number[-1]
    else:
        number = float(str_number)
        magnitude_str = ""

    magnitude = ["", "K", "M", "B", "T"].index(magnitude_str)
    return int(number * 1000**magnitude)


def dict_to_json_dict(dict_to_convert: dict) -> dict:
    """Recursively converge the context of a dictionary to json format"""
    out_dict = {}

    for key, val in dict_to_convert.items():
        if isinstance(val, dict):
            out_dict[key] = dict_to_json_dict(val)
        else:
            try:
                json.dumps(val)
                out_dict[key] = val
            except TypeError:
                out_dict[key] = str(val)

    return out_dict


def append_to_json(json_file_path: str, key: str, val: any) -> None:
    """Append data to an existing json file if it already exists"""
    existing_data = {}
    if os.path.exists(json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as json_file:
            existing_data = json.load(json_file)

    existing_data[key] = val

    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(dict_to_json_dict(existing_data), json_file, indent=4)


def write_to_json(json_file_path: str, data) -> None:
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(dict_to_json_dict(data), json_file, indent=4)


def str_to_act_fn(name: str) -> nn.Module:
    """Converge given string to registered activation function"""

    activation_fn_class = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "LeakyReLU": nn.LeakyReLU,
        "SiLU": nn.SiLU,
    }

    if name not in activation_fn_class:
        raise TypeError(f"{name} is not a recognized activation function.")

    return activation_fn_class[name]


def sigmoid(arr: np.ndarray):
    """Return the sigmoid transform of the input np array"""
    return 1 / (1 + np.exp(-arr))


def get_ci(
    data: np.ndarray,
    confidence: float = 0.95,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    lower_ci, upper_ci = stats.t.interval(
        confidence=confidence,
        df=data.shape[axis] - 1,
        loc=data.mean(axis=axis),
        scale=stats.sem(data, axis=axis),
    )

    return lower_ci, upper_ci
