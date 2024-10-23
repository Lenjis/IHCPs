import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime


def csv_generator(env_var):
    # Define the input and output paths
    input_path = os.path.join(env_var, 'input')
    output_path = os.path.join(env_var, 'output')

    # Initialize the data list
    data = []

    # Iterate over folders in the input path
    for folder_name in sorted(os.listdir(input_path)):
        input_folder = os.path.join(input_path, folder_name)
        output_folder = os.path.join(output_path, folder_name)

        # Check if both input and output folders exist
        if os.path.isdir(input_folder) and os.path.isdir(output_folder):
            data.append(
                {
                'folder_name': folder_name,
                'input_folder': input_folder,
                'output_folder': output_folder
                }
            )

    # Convert the list to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(env_var, 'dataset_info.csv'), index=False)





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




# Function to plot model history from a DataFrame
def plotModelHistoryFromFile_MSE(history_df, fig_initial_value=1, save_fig=False, file_name=None, fig_vars=None):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4))

    fontsize_label = 12
    
    # Extract keys from DataFrame columns
    keys = history_df.columns
    
    # Plot settings: each tuple contains (training variable, validation variable, plot title)
    # Default plot variables if not provided
    if fig_vars is None:
        fig_vars = [
            ('loss', 'val_loss', "Train loss vs Validation loss"),
            ('mean_squared_error', 'val_mean_squared_error', "Train MSE vs Validation MSE")
        ]
    
    # Number of epochs (assuming epochs start from fig_initial_value)
    epochs = range(fig_initial_value, fig_initial_value + len(history_df['loss']))
    
    # Iterate over the plot settings and plot each subplot
    for i, (train_var, val_var, title) in enumerate(fig_vars):
        if train_var in keys and val_var in keys:
            ax[i].semilogy(epochs, history_df[train_var], label=train_var)
            ax[i].semilogy(epochs, history_df[val_var], label=val_var)
            ax[i].legend(fontsize=fontsize_label)
            ax[i].set_title(title, fontsize=fontsize_label)
            
            # Additional settings for the subplot
            for spine in ax[i].spines.values():
                spine.set_linewidth(2)
            ax[i].tick_params(direction='in', which='both')
            ax[i].grid(which='both', linestyle='--', linewidth=0.5)
            ax[i].set_xlabel('Epoch', fontsize=fontsize_label)
            ax[i].set_ylabel(ax[i].get_ylabel(), fontsize=fontsize_label)
        else:
            print(f"Warning: {train_var} or {val_var} not found in DataFrame columns.")

    fig.tight_layout()  # Ensure subplots do not overlap
    plt.show()
    
    # Save the figure if required
    if save_fig:
        if file_name is None:
            # Generate a default file name with a timestamp if none is provided
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f'res_{timestamp}.png'
        fig.savefig(file_name, dpi=300)
        print(f"Figure saved as {file_name}")

    # Print max MSE values
    if 'loss' in keys and 'val_loss' in keys:
        print("Max. Training MSE:", max(history_df['loss']))
        print("Max. Validation MSE:", max(history_df['val_loss']))
    else:
        print("Warning: mean_squared_error or val_mean_squared_error not found in DataFrame columns.")
    
    # Print all history keys and values
    #print("Columns:", keys)
    #for key in keys:
    #    print(f"{key}:", history_df[key].values)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import AutoMinorLocator, LogLocator
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

# Define the path to the Times New Roman font on Windows
#font_properties = FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')

TICK_LABEL_SIZE = 16  # Increased tick label size

# Update rcParams to use Times New Roman for all text elements
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'


#save_path = 'E:\\Project\\cross-sectional project'

def apply_plot_formatting(x_label='x', 
                          y_label='y', 
                          title='title', 
                          font_properties=FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf') # type: ignore
                          ):
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    if title:
        plt.title(title, fontsize=20)
    plt.legend(fontsize=16, 
               frameon=True, 
               framealpha=1, 
               edgecolor='black', 
               fancybox=False,
               borderpad=0.3, labelspacing=0.2
               )
    plt.grid(which='both', linestyle='--', linewidth=1)  # Set grid lines to be dashed

    # Set the linewidth of the x and y axes
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)

    # Enable minor ticks
    plt.gca().minorticks_on()
    # Automatically adjust minor ticks to have one minor tick between each pair of major ticks on x-axis
    plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))


    # Set minor ticks for y-axis on a logarithmic scale
    plt.gca().yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=10))

    # Set font properties for tick labels and make ticks inline
    plt.gca().tick_params(axis='both', which='major', direction='in', length=6, width=1.5, pad=5)
    plt.gca().tick_params(axis='both', which='minor', direction='in', length=3, width=1, pad=5)
    plt.gca().tick_params(axis='x', which='both', top=True, bottom=True)
    plt.gca().tick_params(axis='y', which='both', left=True, right=True)

    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontproperties(font_properties)
        label.set_fontsize(TICK_LABEL_SIZE)  # Set the font size for tick labels
