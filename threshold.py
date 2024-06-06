import cv2
import os
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Create a new folder called "thresholded"
if not os.path.exists("thresholded"):
    os.makedirs("thresholded")

# Only use 2024NewYearLogin-Full.png for now
# filename = "5_5_19_MP4.png"
# filename = "Parako-300.png"
# filename = "poke3.png"
filename = "poke2.png"
# filename = "LastBirdsHope-1.png"
filename_no_ext, _ = os.path.splitext(filename)

img = cv2.imread("heatmap/" + filename, cv2.IMREAD_GRAYSCALE)

# Compute mean value of the image
mean = cv2.mean(img)[0]

# Set threshold value to the middle point between the mean and maximum pixel value
mean = (2 * mean + 255) / 3

# Apply thresholding to each image
_, thresholded = cv2.threshold(img, mean, 255, cv2.THRESH_BINARY)

# Apply morphological operations to remove noise
kernel = np.ones((9, 9), np.uint8)
thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

# Save the thresholded images in the folder "thresholded"
cv2.imwrite("thresholded/" + filename, thresholded)

# Find all connected components in the thresholded image using connectedComponentsWithStats
num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(thresholded)

# Extract coordinates of all white pixels for each connected component
component_coords = []
valid_centroids = []
valid_labels = []  # To store labels of valid components
for i in range(1, num_labels):  # Skip the background component (label 0)
    component_mask = (labels_im == i)
    if np.any(thresholded[component_mask] == 255):  # Check if the component is white
        coords = np.column_stack(np.where(component_mask))
        component_coords.append(coords)
        valid_centroids.append(centroids[i])
        valid_labels.append(i)  # Store the label of the valid component

# Calculate the pairwise minimum distance between connected components
num_components = len(component_coords)
distance_matrix = np.zeros((num_components, num_components))

for i in range(num_components):
    for j in range(i + 1, num_components):
        min_distance = np.min(np.linalg.norm(component_coords[i][:, None] - component_coords[j][None, :], axis=2))
        distance_matrix[i, j] = min_distance
        distance_matrix[j, i] = min_distance

# Perform hierarchical clustering using the distance matrix
condensed_distance_matrix = squareform(distance_matrix)
Z = linkage(condensed_distance_matrix, method='single')

# Choose a threshold to form flat clusters
max_distance = 50  # Adjust this value based on your requirement

clusters = fcluster(Z, max_distance, criterion='distance')

# Convert valid_centroids to a numpy array
valid_centroids = np.array(valid_centroids)

# Map cluster labels back to the original image pixels
clustered_img = np.zeros_like(labels_im, dtype=np.uint8)
for label, cluster in zip(valid_labels, clusters):
    clustered_img[labels_im == label] = cluster

# Create a color map for visualization
colormap = plt.get_cmap('tab20', np.max(clusters) + 1)
colorized_img = colormap(clustered_img / np.max(clusters))  # Normalize for colormap

# Save the colorized image
cv2.imwrite("thresholded/" + filename_no_ext + "_colorized_clusters.png", (colorized_img[:, :, :3] * 255).astype(np.uint8))

# Find the largest cluster, the cluster with the most number of pixels
pixel_counts = np.bincount(clustered_img.flatten())
largest_cluster = np.argmax(pixel_counts[1:]) + 1  # Skip the background (label 0)

# Find the bounding box of the largest cluster, in [left, right, top, bottom] format
largest_cluster_mask = (clustered_img == largest_cluster)
largest_cluster_bbox = cv2.boundingRect(np.column_stack(np.where(largest_cluster_mask)))

# Draw the bounding box on the original image
original_img = cv2.imread("heatmap/" + filename)
x, y, w, h = largest_cluster_bbox
cv2.rectangle(original_img, (y, x), (y + h, x + w), (0, 255, 0), 2)

# Save the original image with bounding box
cv2.imwrite("thresholded/" + filename_no_ext + "_largest_cluster_bbox.png", original_img)

print(f"Bounding box coordinate: [{y}, {y + h}, {x}, {x + w}]")
print(f"Bounding box coordinate ratio: [{y / img.shape[1]:.2f}, {(y + h) / img.shape[1]:.2f}, {x / img.shape[0]:.2f}, {(x + w) / img.shape[0]:.2f}]")
