import numpy as np
import cv2

from motion_estimation._2D.farneback_OpenCV import OF_Estimation
from motion_estimation._2D.project import Projection

import logging
logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

estimator = OF_Estimation(logger)
projector = Projection(logger)

def shake(x, y, std_dev=1.0):
    """
    Apply Gaussian noise to the given indices and return a sorted mapping.
    """
    displacements = np.random.normal(loc=0, scale=std_dev, size=len(x))
    return np.stack((y + displacements, x), axis=1)

def randomize_image(image, std_dev=16.0):
    """
    Apply a controlled randomization to a 2D image.
    This perturbs pixel positions along the X and Y axes while maintaining structures.
    """
    randomized_image = np.empty_like(image)
    
    # Randomization in X (Columns)
    values = np.arange(image.shape[1])  # Column indices
    for y in range(image.shape[0]):  # Iterate over rows
        pairs = shake(values, np.arange(len(values)), std_dev).astype(np.int32)
        pairs = pairs[pairs[:, 0].argsort()]  # Sort based on perturbed positions
        randomized_image[y, values] = image[y, pairs[:, 1]]  # Apply reordering

    # Randomization in Y (Rows)
    values = np.arange(image.shape[0])  # Row indices
    for x in range(image.shape[1]):  # Iterate over columns
        pairs = shake(values, np.arange(len(values)), std_dev).astype(np.int32)
        pairs = pairs[pairs[:, 0].argsort()]  # Sort based on perturbed positions
        randomized_image[values, x] = randomized_image[pairs[:, 1], x]  # Apply reordering

    return randomized_image

def project_A_to_B(A, B, window_side=5, sigma_poly=1.2):
  zeros = np.zeros((A.shape[0], A.shape[1], 2), dtype=np.float32)
  MVs = estimator.pyramid_get_flow(
      target=A, reference=B,
      flow=zeros,
      window_side=window_side,
      sigma_poly=sigma_poly)
  try:
    projection = projector.remap(A, MVs)
  except cv2.error:
    return A
  return projection

def randomize_and_project(image, std_dev=3.0, window_side=5, sigma_poly=1.2):
  randomized_image = randomize_image(image, std_dev)
  projection = project_A_to_B(
      A=image,
      B=randomized_image, # Ojo, pueden estar al revés
      window_side=window_side,
      sigma_poly=sigma_poly) 
  return projection
  #return randomized_image

def randomize_and_project3(image, std_dev=3.0, window_side=5, sigma_poly=1.2):
  randomized_image = randomize_image(image, std_dev)
#  projection = project_A_to_B(A=image, B=randomized_image, window_side=window_side, sigma_poly=sigma_poly) # Ojo, pueden estar al revés
  #return projection
  return randomized_image

def randomize_and_project2(image, std_dev=3.0, window_side=5, sigma_poly=1.2):
  randomized_image = randomize_image(image, std_dev)
  projection = project_A_to_B(B=image, A=randomized_image, window_side=window_side, sigma_poly=sigma_poly) # Ojo, pueden estar al revés
  return projection
  #return randomized_image

def chessboard_interpolate_blacks(image):
    height, width = image.shape[:2]
    output_image = image.copy()

    y, x = np.mgrid[:height, :width]
    mask = (x + y) % 2 != 0

    # Create shifted versions of the image for neighbors
    if image.ndim == 3:  # Color image
        left = np.pad(image[:, :-1], ((0, 0), (1, 0), (0, 0)), mode='edge')
        right = np.pad(image[:, 1:], ((0, 0), (0, 1), (0, 0)), mode='edge')
        up = np.pad(image[:-1, :], ((1, 0), (0, 0), (0, 0)), mode='edge')
        down = np.pad(image[1:, :], ((0, 1), (0, 0), (0, 0)), mode='edge')
        #upleft = np.pad(image[:-1, :-1],((1,0),(1,0),(0,0)), mode='edge')
        #upright = np.pad(image[:-1, 1:],((1,0),(0,1),(0,0)), mode='edge')
        #downleft = np.pad(image[1:, :-1],((0,1),(1,0),(0,0)), mode='edge')
        #downright = np.pad(image[1:, 1:],((0,1),(0,1),(0,0)), mode='edge')
    else:  # Grayscale image
        left = np.pad(image[:, :-1], ((0, 0), (1, 0)), mode='edge')
        right = np.pad(image[:, 1:], ((0, 0), (0, 1)), mode='edge')
        up = np.pad(image[:-1, :], ((1, 0), (0, 0)), mode='edge')
        down = np.pad(image[1:, :], ((0, 1), (0, 0)), mode='edge')
        #upleft = np.pad(image[:-1, :-1],((1,0),(1,0)), mode='edge')
        #upright = np.pad(image[:-1, 1:],((1,0),(0,1)), mode='edge')
        #downleft = np.pad(image[1:, :-1],((0,1),(1,0)), mode='edge')
        #downright = np.pad(image[1:, 1:],((0,1),(0,1)), mode='edge')

    # Calculate the average of neighbors for "black" pixels
    #neighbors_avg = np.mean(np.stack([left, right, up, down, upleft, upright, downleft, downright], axis=-1), axis=-1)
    neighbors_avg = np.mean(np.stack([left, right, up, down], axis=-1), axis=-1)

    # Apply the interpolated values to the "black" pixels
    output_image[mask] = neighbors_avg[mask]
    #output_image[mask] = 0

    return output_image

def chessboard_interpolate_whites(image):
    height, width = image.shape[:2]
    output_image = image.copy()

    y, x = np.mgrid[:height, :width]
    mask = (x + y) % 2 == 0

    # Create shifted versions of the image for neighbors
    if image.ndim == 3:  # Color image
        left = np.pad(image[:, :-1], ((0, 0), (1, 0), (0, 0)), mode='edge')
        right = np.pad(image[:, 1:], ((0, 0), (0, 1), (0, 0)), mode='edge')
        up = np.pad(image[:-1, :], ((1, 0), (0, 0), (0, 0)), mode='edge')
        down = np.pad(image[1:, :], ((0, 1), (0, 0), (0, 0)), mode='edge')
        #upleft = np.pad(image[:-1, :-1],((1,0),(1,0),(0,0)), mode='edge')
        #upright = np.pad(image[:-1, 1:],((1,0),(0,1),(0,0)), mode='edge')
        #downleft = np.pad(image[1:, :-1],((0,1),(1,0),(0,0)), mode='edge')
        #downright = np.pad(image[1:, 1:],((0,1),(0,1),(0,0)), mode='edge')
    else:  # Grayscale image
        left = np.pad(image[:, :-1], ((0, 0), (1, 0)), mode='edge')
        right = np.pad(image[:, 1:], ((0, 0), (0, 1)), mode='edge')
        up = np.pad(image[:-1, :], ((1, 0), (0, 0)), mode='edge')
        down = np.pad(image[1:, :], ((0, 1), (0, 0)), mode='edge')
        #upleft = np.pad(image[:-1, :-1],((1,0),(1,0)), mode='edge')
        #upright = np.pad(image[:-1, 1:],((1,0),(0,1)), mode='edge')
        #downleft = np.pad(image[1:, :-1],((0,1),(1,0)), mode='edge')
        #downright = np.pad(image[1:, 1:],((0,1),(0,1)), mode='edge')

    # Calculate the average of neighbors for "white" pixels
    #neighbors_avg = np.mean(np.stack([left, right, up, down, upleft, upright, downleft, downright], axis=-1), axis=-1)
    neighbors_avg = np.mean(np.stack([left, right, up, down], axis=-1), axis=-1)

    # Apply the interpolated values to the "white" pixels
    output_image[mask] = neighbors_avg[mask]
    #output_image[mask] = 0
    return output_image

def chessboard_blacks(image):
    height, width = image.shape[:2]
    output_image = image.copy()

    y, x = np.mgrid[:height, :width]
    mask = (x + y) % 2 != 0
    output_image[mask] = 0

    return output_image

def chessboard_whites(image):
    height, width = image.shape[:2]
    output_image = image.copy()

    y, x = np.mgrid[:height, :width]
    mask = (x + y) % 2 == 0
    output_image[mask] = 0
    return output_image

def chessboard(image):
    height, width = image.shape[:2]
    blacks_image = image.copy()
    whites_image = image.copy()
    y, x = np.mgrid[:height, :width]
    blacks_mask = (x + y) % 2 != 0
    whites_mask = (x + y) % 2 == 0
    blacks_image[blacks_mask] = 0
    whites_image[whites_mask] = 0
    return blacks_image, whites_image

def subsampled_chessboard(image):
    '''Takes an image as input and splits it into four sub-images
    based on a chessboard-like subsampling pattern'''
    # https://www.nature.com/articles/s41467-019-11024-z
    # https://github.com/sakoho81/miplib/blob/public/miplib/processing/image.py#L133
    shape = image.shape
    odd_index = list(np.arange(1, shape[i], 2) for i in range(len(shape))) # There are two lists because the imagen can be rectangular
    even_index = list(np.arange(0, shape[i], 2) for i in range(len(shape)))
    odd_odd = image[odd_index[0], :][:, odd_index[1]]
    even_even = image[even_index[0], :][:, even_index[1]]
    odd_even = image[odd_index[0], :][:, even_index[1]]
    even_odd = image[even_index[0], :][:, odd_index[1]]
    return odd_odd, even_even, odd_even, even_odd

import numpy as np

def fade_image_margins(image_array, fade_width=0):
    """
    Smooths the margins of a grayscale image (NumPy array with float pixels)
    by fading the intensity of the margin pixels towards zero.

    Args:
        image_array (np.ndarray): A NumPy array representing the grayscale image
                                    (shape: (height, width)) with float pixel values
                                    (e.g., between 0.0 and 1.0 or 0.0 and 255.0).
        fade_width_pixels (int): The width of the fade effect from the edge inwards, in pixels.

    Returns:
        np.ndarray: A NumPy array representing the grayscale image with faded margins (float pixels).
    """
    height, width = image_array.shape

    modified_array = image_array.astype(float)  # Convert to float for precise calculations

    # Create Gaussian kernel for fading (1D)
    x = np.linspace(0, 1, fade_width)
    fade_curve = np.exp(-(x)**2)  # Adjust multiplier for sharper/softer fade

    # Apply fading to top margin
    for y in range(fade_width):
        weight = fade_curve[fade_width - 1 - y]
        #modified_array[y, :] *= (1 - weight)
        modified_array[y, :] *= weight

    # Apply fading to bottom margin
    for y in range(fade_width):
        weight = fade_curve[fade_width - y - 1]
        #modified_array[height - 1 - y, :] *= (1 - weight)
        modified_array[height - 1 - y, :] *= weight

    # Apply fading to left margin
    for x in range(fade_width):
        weight = fade_curve[fade_width - 1 - x]
        #modified_array[:, x] *= (1 - weight)
        modified_array[:, x] *= weight

    # Apply fading to right margin
    for x in range(fade_width):
        weight = fade_curve[fade_width - x - 1]
        #modified_array[:, width - 1 - x] *= (1 - weight)
        modified_array[:, width - 1 - x] *= weight

    return modified_array.astype(np.uint8)  # Convert back to uint8




