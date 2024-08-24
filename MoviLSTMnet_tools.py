import numpy as np


def cal_true_value(heatflux_type, groundtruth_heatflux=None, closest_filename_index=0, time_point=None, A=None):
    """
    Calculate the true value of heat flux based on the specified type.

    Parameters:
    heatflux_type (str): Type of heat flux ('constant' or 'sin').
    groundtruth_heatflux (list, optional): List of ground truth heat flux values. Required if heatflux_type is 'constant'.
    closest_filename_index (int): Index for selecting the ground truth value.
    time_point (float, optional): Time point for sine function calculation. Required if heatflux_type is 'sin'.
    A (float, optional): Amplitude for sine function. Required if heatflux_type is 'sin'.

    Returns:
    float: Calculated true value of heat flux.
    """

    if heatflux_type == 'constant':
        if groundtruth_heatflux is None:
            raise ValueError("groundtruth_heatflux must be provided when heatflux_type is 'constant'.")
        # If the heat flux is constant, return the ground truth value directly
        true_value = groundtruth_heatflux[closest_filename_index]
    elif heatflux_type == 'sin':
        if time_point is None or A is None:
            raise ValueError("For 'sin' type, both 'time_point' and 'A' must be provided.")
        # Calculate the true value using the sine function
        true_value = A * np.sin(((np.pi * time_point + 1) / 30) - np.pi / 2) + A
    else:
        raise ValueError("Invalid heatflux_type. Must be 'constant' or 'sin'.")

    return true_value
