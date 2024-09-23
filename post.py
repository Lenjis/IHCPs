import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the ground truth image
gt_image_path = 'result_sin/ground_truth/ground_truth_0.png'
gt_image = Image.open(gt_image_path)
gt_array = np.array(gt_image)

# Load the predicted image
pred_image_path = 'result_sin/predicted_frames/predicted_frame_0.png'
pred_image = Image.open(pred_image_path)
pred_array = np.array(pred_image)

# Ensure the images have the same shape
assert gt_array.shape == pred_array.shape, "The ground truth and predicted images must have the same shape."

# Calculate the errors
errors = pred_array - gt_array

# Calculate the magnitude of the errors
error_magnitude = np.abs(errors)

# Extract x and y coordinates
height, width = gt_array.shape[:2]
x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
x_coords = x_coords.flatten()
y_coords = y_coords.flatten()
error_magnitude = error_magnitude.flatten()


# Assuming x_coords, y_coords, and error_magnitude are already defined
if len(x_coords) != len(y_coords) or len(x_coords) != len(error_magnitude):
    raise ValueError("x_coords, y_coords, and error_magnitude must all have the same length")

# Create a scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(x_coords, y_coords, c=error_magnitude, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Error Magnitude')
plt.title('Error Distribution Based on XY Location')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.show()
