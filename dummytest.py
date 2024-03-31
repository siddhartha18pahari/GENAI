import numpy as np
import matplotlib.pyplot as plt

def generate_dummy_data(batch_size=9):
    """
    Generates dummy image data and labels for demonstration purposes.
    
    Parameters:
    - batch_size (int): Number of images and labels to generate.
    
    Returns:
    - images (numpy.ndarray): Array of image data.
    - true_labels (numpy.ndarray): Array of true labels.
    - predicted_labels (numpy.ndarray): Array of predicted labels.
    """
    # Generate random images
    images = np.random.rand(batch_size, 224, 224, 3)
    
    # Generate true labels (binary) and then randomly flip some for predicted labels
    true_labels = np.random.randint(2, size=batch_size)
    predicted_labels = true_labels.copy()
    
    # Introduce errors in predictions for a third of the batch
    error_indices = np.random.choice(batch_size, size=batch_size // 3, replace=False)
    predicted_labels[error_indices] = 1 - predicted_labels[error_indices]
    
    return images, true_labels, predicted_labels

def visualize_predictions(images, true_labels, predicted_labels):
    """
    Visualizes images with their true and predicted labels.
    
    Parameters:
    - images (numpy.ndarray): Array of image data.
    - true_labels (numpy.ndarray): Array of true labels.
    - predicted_labels (numpy.ndarray): Array of predicted labels.
    """
    batch_size = len(images)
    nrows = int(np.ceil(batch_size / 3))
    ncols = 3 if batch_size > 2 else batch_size
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 2.5 * nrows),
                            subplot_kw={'xticks': [], 'yticks': []})
    
    # Flatten the axis array for easy iteration if there's more than one row
    if axs.ndim > 1:
        axs = axs.flatten()
        
    for i, ax in enumerate(axs):
        if i < batch_size:  # Check to avoid error when batch_size is not a multiple of ncols
            ax.imshow(images[i])
            title_color = 'green' if true_labels[i] == predicted_labels[i] else 'red'
            ax.set_title(f"True: {true_labels[i]}, Pred: {predicted_labels[i]}", color=title_color)
        ax.axis('off')  # Hide axes for empty plots
    
    plt.tight_layout()
    plt.show()

# Example usage
batch_size = 9
images, true_labels, predicted_labels = generate_dummy_data(batch_size)
visualize_predictions(images, true_labels, predicted_labels)
