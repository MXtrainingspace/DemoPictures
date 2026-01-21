import cv2

# Bild einlesen
image = cv2.imread('../testpicture_1.png')
# In Graustufen umwandeln
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Bildrauschen reduzieren
blur_image = cv2.GaussianBlur(gray_image,(5,5),0)
# Canny-Kantenerkennung anwenden
canny_image = cv2.Canny(blur_image, 50, 150)
# Ergebnis anzeigen
cv2.imshow("result", canny_image)
cv2.waitKey(0)