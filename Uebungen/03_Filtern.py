import matplotlib.pyplot as plt
import cv2
import numpy as np

image_path='../testpicture_1.png'

# Filterelement nach Aufgabenstellung definieren
gauss = [[1/273, 4/273, 7/273, 4/273, 1/273],
 [4/273, 16/273, 26/273, 16/273, 4/273],
  [7/273, 26/273, 41/273, 26/273, 7/273],
   [4/273, 16/273, 26/273, 16/273, 4/273],
   [1/273, 4/273, 7/273, 4/273, 1/273]]


def filter_image(filter, image):
  f_h, f_w = len(filter), len(filter[0])
  img_h, img_w = len(image), len(image[0])
  f_r_offset = list(range(-f_h // 2+1, f_h // 2 + 1))
  f_w_offset = list(range(-f_w // 2+1, f_w // 2 + 1))
  output = []

  for row in range(img_h):
    new_row = []
    for col in range(img_w):
      sum = 0
      for f_r in f_r_offset:
        for f_c in f_w_offset:
          curr_row, curr_col = row+f_r, col+f_c
          if(curr_row > 0) and (curr_row < img_h) and (curr_col > 0) and (curr_col < img_w):
            sum += filter[f_r][f_c]*image[curr_row][curr_col]
      new_row.append(sum)
    output.append(new_row)

  return output

img = cv2.imread(image_path)
img_grau = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bw_lst = img_grau.tolist()
smooth = filter_image(gauss, bw_lst)


plt.figure()
plt.imshow(bw_lst, cmap='gray')
plt.axis('off')
plt.title('Graustufenbild')
plt.figure()
plt.imshow(smooth, cmap='gray')
plt.axis('off')
plt.title('Gefiltert mit einem 5x5 Gaußfilter')
plt.show()

def filter_image_np(filter, image):
    filter = np.matrix(filter)
    h,w = image.shape
    new_img = []
    for row in range(2, h-2):
        new_row = []
        for col in range(2, w-2):
            neighbors = image[row-2:row+3, col-2:col+3]
            product = np.multiply(neighbors, filter)
            value = np.sum(product)    
            new_row.append(int(value))
        new_img.append(new_row)
    return new_img

np_version = filter_image_np(gauss, img_grau)

plt.figure()
plt.imshow(np_version, cmap='gray')
plt.axis('off')
plt.title('Gefiltert mit einem 5x5 Gaußfilter mit NumPy')
plt.show()