import os

import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch

import io
import pstats
import cProfile
from pstats import SortKey
import torch
from torch.profiler import profile, record_function, ProfilerActivity


def get_mean_node_activations(input_dict: dict) -> dict:
    output_dict = {}
    for kk in input_dict:
        output_dict_layer = []
        for node in input_dict[kk].T:
            output_dict_layer.append(torch.mean(node).item())
        output_dict[kk] = output_dict_layer
    return output_dict


def dict_to_square_matrix(input_dict: dict) -> np.array:
    """Function changes an input dictionary into a square np.array. Adds NaNs when the dimension of a dict key is less than of the final square matrix.

    Args:
        input_dict (dict)

    Returns:
        square_matrix (np.array)
    """
    means_dict = get_mean_node_activations(input_dict)
    max_number_of_nodes = 0
    number_of_layers = len(input_dict)
    for kk in means_dict:
        if len(means_dict[kk]) > max_number_of_nodes:
            max_number_of_nodes = len(means_dict[kk])
    square_matrix = np.empty((number_of_layers, max_number_of_nodes))
    counter = 0
    for kk in input_dict:
        layer = np.array(means_dict[kk])
        if len(layer) == max_number_of_nodes:
            square_matrix[counter] = layer
        else:
            layer = np.append(
                layer, np.zeros(max_number_of_nodes - len(layer)) + np.nan
            )
            square_matrix[counter] = layer
        counter += 1
    return square_matrix


def plot(data: np.array, output_path: str) -> None:
    nodes_numbers = np.array([0, 50, 100, 200])
    fig, ax = plt.subplots()
    NAP = ax.imshow(
        data.T,
        cmap="RdBu_r",
        interpolation="nearest",
        aspect="auto",
        origin="lower",
        norm=matplotlib.colors.CenteredNorm(),
    )
    colorbar = plt.colorbar(NAP)
    colorbar.set_label("Activation")
    ax.set_title("Neural Activation Pattern")
    ax.set_xlabel("Layers")
    ax.set_ylabel("Number of nodes")
    xtick_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(xtick_loc))
    ax.set_xticklabels(["", "en1", "en2", "en3", "de1", "de2", "de3", ""])
    ax.set_yticks(nodes_numbers)
    ax.figure.savefig(os.path.join(output_path, "diagnostics.pdf"))


def nap_diagnose(input_path: str, output_path: str) -> None:
    input = np.load(input_path)
    plot(input, output_path)
    print(
        "Diagnostics saved as diagnostics.pdf in the diagnostics folder of your project."
    )


def pytorch_profile(f, *args, **kwargs):
    """
    This function performs PyTorch profiling of CPU, GPU time and memory
    consumed by the function f execution.

    Args:
        f (callable): The function to be profiled.

    Returns:
        result: The result of the function `f` execution.
    """

    if torch.cuda.is_available():
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    else:
        activities = [ProfilerActivity.CPU]

    # Start profiler before the function will be executed
    with profile(
        activities=activities,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "log/baler", worker_name="worker0"
        ),
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        with record_function(f"{f.__name__}"):
            # Call the function
            result = f(*args, **kwargs)
            prof.step()
            prof.stop()

    # Print the CPU time for each torch operation
    print(prof.key_averages().table(sort_by="cpu_time_total"))

    # Store the information about CPU and GPU usage
    if torch.cuda.is_available():
        prof.export_stacks("profiler_stacks.json", "self_cuda_time_total")

    # Store the results to the .json file
    prof.export_stacks("/tmp/profiler_stacks.json", "self_cpu_time_total")

    return result


def c_profile(func, *args, **kwargs):
    """
    Profile the function func with cProfile.

    Args:
        func (callable): The function to be profiled.

    Returns:
        result: The result of the function `func` execution.
    """

    pr = cProfile.Profile()
    pr.enable()
    # Execute the function and get its result
    result = func(*args, **kwargs)
    pr.disable()

    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    return result
