import cv2
import matplotlib.pyplot as plt

image_path='../testpicture_1.png'
cv2_img = cv2.imread(image_path)
plt_img = plt.imread(image_path)
cv2_rgb_np = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
img_list = cv2_rgb_np.tolist()

cv2.imshow("OpenCV Window", cv2_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.imshow(plt_img)
plt.axis('off')
plt.title("Version mit Matplotlib angezeigt")
plt.show()
plt.imshow(img_list)
plt.title("Version mit cv2 -> Liste angezeigt")
plt.show()
plt.close()

# Laenge des Bildes berechnen
reihen = len(img_list)

# Breite des Bildes berechnen
spalten = len(img_list[0]) #img_list[0] ist die erste Reihe im Bild

haelfte_reihen = reihen//2 #Abrundung von Division reihen durch 2

bottom_image = img_list[haelfte_reihen:][:][:] #Slicing: Alle Reihen ab Haelfte,
                                                        #Alle Spalten
                                                        #Alle Fahrben

plt.imshow(bottom_image)
plt.axis('off')
plt.title("Untere Haelfte")
plt.show()
