import matplotlib.pyplot as plt
import cv2

image_path='../testpicture_1.png'
img_list = plt_img = plt.imread(image_path)

img = cv2.imread(image_path)
dims = img.shape
height, width, channels = dims
new_height = height//2
new_img = img[new_height:,:,:]
new_img_list = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB).tolist()

# OpenCV-Graustufen
new_img_bw = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
new_img_bw_list = new_img_bw.tolist()

rot_gewicht = 0.299
gruen_gewicht = 0.587
blau_gewicht = 0.114

# Manuelle Graustufen
height, width, channels = new_img.shape
bw_lst = []
for row in range(height):
    curr_row = []
    for col in range(width):
        B = new_img[row, col, 0]
        G = new_img[row, col, 1]
        R = new_img[row, col, 2]
        curr_row.append(rot_gewicht * R + gruen_gewicht * G + blau_gewicht * B)
    bw_lst.append(curr_row)

# Erste Darstellung
plt.figure()
plt.imshow(new_img_bw_list, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
plt.title('OpenCV Graustufen')
plt.axis('off')
plt.show()

# Zweite Darstellung
plt.figure()
plt.imshow(bw_lst, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
plt.title('Selber berechnet')
plt.axis('off')
plt.show()