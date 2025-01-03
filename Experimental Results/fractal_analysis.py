import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def calculate_box_count(image, box_sizes):
    """Calculate the number of boxes required to cover the fractal."""
    counts = []
    for box_size in box_sizes:
        # Resize image to make it divisible by box size
        new_size = (np.ceil(np.array(image.shape) / box_size) * box_size).astype(int)
        padded_image = np.zeros(new_size, dtype=image.dtype)
        padded_image[:image.shape[0], :image.shape[1]] = image
        
        # Count non-empty boxes
        reshaped = padded_image.reshape(
            new_size[0] // box_size, box_size,
            new_size[1] // box_size, box_size
        )
        box_sum = reshaped.sum(axis=(1, 3))
        counts.append(np.sum(box_sum > 0))
    return counts

def estimate_fractal_dimension(image_path, min_box_size=2, max_box_size=128):
    """Estimate the fractal dimension of a binary fractal image."""
    # Load and preprocess the image
    image = Image.open(image_path).convert('L') # Convert to grayscale
    binary_image = np.array(image) > 128 # Convert to binary (thresholding)
    
    # Generate box sizes
    box_sizes = np.logspace(
        np.log2(min_box_size), np.log2(max_box_size),
        num=20, base=2, dtype=int
    )
    box_sizes = np.unique(box_sizes) # Remove duplicates
    
    # Calculate box counts
    counts = calculate_box_count(binary_image, box_sizes)
    
    # Perform linear regression on log-log data
    log_sizes = np.log(1 / box_sizes)
    log_counts = np.log(counts)
    
    # Exclude invalid counts
    valid = np.isfinite(log_counts)
    log_sizes = log_sizes[valid]
    log_counts = log_counts[valid]
    
    # Fit a line to log-log data
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dimension = coeffs[0]
    
    # Plot results
    plt.figure()
    plt.plot(log_sizes, log_counts, 'o', label='Data')
    plt.plot(log_sizes, np.polyval(coeffs, log_sizes), '-', label=f'Fit (D={fractal_dimension:.3f})')
    plt.xlabel('log(1/box size)')
    plt.ylabel('log(box count)')
    plt.legend()
    plt.title('Fractal Dimension Estimation')
    plt.show()
    
    return fractal_dimension

if __name__ == '__main__':
    directory = 'Experimental Results/branches/'
    images = sorted(os.listdir(directory))
    for image_path in images:
        fractal_dimension = estimate_fractal_dimension(directory + image_path)
        print(f"Estimated Fractal Dimension for {image_path}: {fractal_dimension:.3f}")