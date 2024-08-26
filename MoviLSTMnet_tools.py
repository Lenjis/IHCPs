import numpy as np


def cal_true_value(heatflux_type, **kwargs):
    """
    Calculate the true value of heat flux based on the specified type.

    Parameters:
        heatflux_type (str): Type of heat flux ('constant' or 'sin').
        kwargs: Additional parameters based on the heatflux_type.

            For 'constant':
                groundtruth_heatflux (list): List of ground truth heat flux values.
                closest_filename_index (int): Index for selecting the ground truth value.
            For 'sin':
                time_point (float): Time point for sine function calculation.
                A (float): Amplitude for sine function.
            

                
    Returns:
        float: Calculated true value of heat flux.
    
    """
    if heatflux_type == 'constant':
        groundtruth_heatflux = kwargs.get('groundtruth_heatflux')
        closest_filename_index = kwargs.get('closest_filename_index', 0)

        if groundtruth_heatflux is None:
            raise ValueError("groundtruth_heatflux must be provided when heatflux_type is 'constant'.")

        # If the heat flux is constant, return the ground truth value directly
        true_value = groundtruth_heatflux[closest_filename_index]

    elif heatflux_type == 'sin':
        time_point = kwargs.get('time_point')
        A = kwargs.get('A')

        if time_point is None or A is None:
            raise ValueError("For 'sin' type, both 'time_point' and 'A' must be provided.")

        # Calculate the true value using the sine function
        true_value = A * np.sin(((np.pi * time_point + 1) / 30) - np.pi / 2) + A

    else:
        raise ValueError("Invalid heatflux_type. Must be 'constant' or 'sin'.")

    return true_value

