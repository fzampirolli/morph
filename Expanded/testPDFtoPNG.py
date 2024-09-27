from pdf2image import convert_from_path
import numpy as np
from PIL import Image
from skimage import morphology, color, transform, filters
from skimage.util import img_as_ubyte

# Upload the PDF file
filename = 'fig_Maze3.pdf'

# Convert PDF to a list of PIL images
images = convert_from_path(filename, dpi=600)

# Process each page
for i, image in enumerate(images):
    # Convert PIL Image to NumPy array
    img_array = np.array(image)

    # Get image dimensions
    height, width = img_array.shape[:2]

    # Define crop coordinates
    top = int(0.06 * height)      # 6% of height
    bottom = int(0.615 * height)  # 61.5% of height
    left = int(0.09 * width)      # 9% of width
    right = int(0.91 * width)     # 91% of width

    # Crop the image
    cropped_img_array = img_array[top:bottom, left:right]
    height, width = cropped_img_array.shape[:2]
    print(cropped_img_array.shape)  # Debug: Print the shape of the array

    # Resize the image to reduce dimensions by a factor of 15
    reduced_height = height // 7
    reduced_width = width // 7
    resized_img_array = transform.resize(cropped_img_array, (reduced_height, reduced_width), anti_aliasing=True)

    print(resized_img_array.shape)  # Debug: Print the shape of the array

    # Convert resized NumPy array to grayscale if it is RGB
    if len(resized_img_array.shape) == 3:
        resized_img_array = color.rgb2gray(resized_img_array)  # Convert to grayscale

    # Binarize the grayscale image using Otsu's method
    threshold = filters.threshold_otsu(resized_img_array)
    binary_img_array = resized_img_array > threshold

    H, W = binary_img_array.shape
    binary_img_array2 = binary_img_array[H//2+15:H, 0:W//2+20]
    print(binary_img_array2.shape)

    H, W = binary_img_array2.shape
    binary_img_array3 = binary_img_array2[150:H, 75:275]
    print(binary_img_array3.shape)

    # Convert the inverted skeleton to 8-bit format for saving
    skeleton_img_a = Image.fromarray(img_as_ubyte(cropped_img_array))
    skeleton_img_b = Image.fromarray(img_as_ubyte(binary_img_array))
    skeleton_img_c = Image.fromarray(img_as_ubyte(binary_img_array2))
    skeleton_img_d = Image.fromarray(img_as_ubyte(binary_img_array3))

    # Save the final processed image
    skeleton_img_a.save(f'fig_Maze3a.png')
    skeleton_img_b.save(f'fig_Maze3b.png')
    skeleton_img_c.save(f'fig_Maze3c.png')
    skeleton_img_d.save(f'fig_Maze3d.png')
