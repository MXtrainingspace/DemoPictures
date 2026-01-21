import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

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


image_path='../testpicture_1.png'
img = cv2.imread(image_path)
img = img[img.shape[0]//2:,:,:]
img_grau = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smooth = filter_image(gauss, img_grau)

# Funktion die den Sobel Operator in Abhängigkeit des Übergabearguments zurück gibt
def sobel(horizontal):
  if horizontal: # oder if(horizontal == true)
    return [[1,2,1],[0,0,0],[-1,-2,-1]]
  else:
    return [[-1,0,1],[-2,0,2],[-1,0,1]]
  
vert = filter_image(sobel(False), smooth)
horz = filter_image(sobel(True), smooth)

def gradient(vert, horz):
  img_h, img_w = len(vert), len(vert[0])

  grad = []
  for row in range(img_h):
    grad_row = []
    for col in range(img_w):
      grad_row.append( math.sqrt(vert[row][col]**2 + horz[row][col]**2)) # oder mit der Methode hypot() berechnen
    grad.append(grad_row)
  return grad

def quantize(angle):
  angle = angle * 180 / math.pi
  if angle >=0 and angle <= 22.5:
    return 0
  elif angle >= 22.5 and angle <= 67.5:
    return 45
  elif angle >= 67.5 and angle <= 112.5:
    return 90
  elif angle >= 112.5 and angle <= 157.5:
    return 135
  else:
    return 0
  
def gradient_winkel(vert, horz):
  img_h, img_w = len(vert), len(vert[0])

  grad_winkel = []
  for row in range(img_h):
    grad_winkel_row = []
    for col in range(img_w):
      angle = math.atan2(vert[row][col], horz[row][col])
      grad_winkel_row.append(quantize(angle))
    grad_winkel.append(grad_winkel_row)
  return grad_winkel

def normalize(img):
  img_h, img_w = len(img), len(img[0])
  max = float('-inf')
  min = float('inf')
  for row in range(img_h):
    for col in range(img_w):
      if img[row][col] > max:
        max = img[row][col]
      if img[row][col] < min:
        min = img[row][col]
  print(f"max value is {max:.2f}, min value is {min:.2f}")
  norm = []
  scale = 255/(max-min)
  for row in range(img_h):
    norm_row = []
    for col in range(img_w):
      norm_row.append( (img[row][col] - min) * scale)
    norm.append(norm_row)

  return norm

g = gradient(vert, horz)
g_w = gradient_winkel(vert, horz)

plt.figure()
plt.imshow(g)
plt.title("Gradientenstärke")
plt.axis("off")

norm = normalize(g)

plt.figure()
plt.imshow(norm,cmap='gray_r')
plt.title("Gradientenstärke normalisiert auf 256 Graustufen\nFarben invertiert")
plt.axis("off")

plt.figure()
plt.imshow(g_w, cmap='gray')
plt.title("Gradientenwinkel")
plt.axis("off")

plt.show()