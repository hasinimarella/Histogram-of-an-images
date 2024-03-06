# Histogram-of-an-images
## Aim
To obtain a histogram for finding the frequency of pixels in an Image with pixel values ranging from 0 to 255. Also write the code using OpenCV to perform histogram equalization.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Read the gray and color image using imread().
### Step2:
Print the image using imshow().
### Step3:
Use calcHist() function to mark the image in graph frequency for gray and color image.

### step4:
Use calcHist() function to mark the image in graph frequency for gray and color image.

### Step5:
The Histogram of gray scale image and color image is shown.


## Program:
```python
# Developed By: MARELLA HASINI
# Register Number: 212223240083
i)
from google.colab import files
import cv2
import matplotlib.pyplot as plt
def display_image(image, title):
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()
uploaded_grayscale = files.upload()
uploaded_color = files.upload()
grayscale_img_path = list(uploaded_grayscale.keys())[0]
grayscale_img = cv2.imread(grayscale_img_path, cv2.IMREAD_GRAYSCALE)
display_image(grayscale_img, 'Grayscale Image')
# Read and display color image
color_img_path = list(uploaded_color.keys())[0]
color_img = cv2.imread(color_img_path)
color_img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB) 
display_image(color_img_rgb, 'Color Image')

ii)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def plot_histogram(image, title):
    plt.hist(image.ravel(), bins=256, range=[0, 256], color='blue')
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.show()
uploaded_grayscale = files.upload()
uploaded_color = files.upload()

# Read grayscale image
grayscale_img_path = list(uploaded_grayscale.keys())[0]
grayscale_img = cv2.imread(grayscale_img_path, cv2.IMREAD_GRAYSCALE)

# Read color image
color_img_path = list(uploaded_color.keys())[0]
color_img = cv2.imread(color_img_path)

# Extract blue channel from color image
blue_channel_img = color_img[:, :, 0]

# Plot histogram of grayscale image
plot_histogram(grayscale_img, 'Histogram of Grayscale Image')

# Plot histogram of blue channel of color image
plot_histogram(blue_channel_img, 'Histogram of Blue Channel of Color Image')

iii)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Function to perform histogram equalization
def histogram_equalization(image):
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

# Upload grayscale image
uploaded_grayscale = files.upload()

# Read grayscale image
grayscale_img_path = list(uploaded_grayscale.keys())[0]
grayscale_img = cv2.imread(grayscale_img_path, cv2.IMREAD_GRAYSCALE)

# Perform histogram equalization
equalized_img = histogram_equalization(grayscale_img)

# Plot original and equalized images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(grayscale_img, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_img, cmap='gray')
plt.title('Histogram Equalized Image')
plt.axis('off')

plt.show()
```
## Output:
### Input Grayscale Image and Color Image
![OUTPUT](<image input.png>)
![OUTPUT](<input 2.png>)

### Histogram of Grayscale Image and any channel of Color Image

![OUTPUT](<input 2.1.png>)
![OUTPUT](<input 2.3.png>)
### Histogram Equalization of Grayscale Image.
![OUTPUT](<input 3.1.png>)
![OUTPUT](<input 3.2.png>)

## Result: 
Thus the histogram for finding the frequency of pixels in an image with pixel values ranging from 0 to 255 is obtained. Also,histogram equalization is done for the gray scale image using OpenCV.
