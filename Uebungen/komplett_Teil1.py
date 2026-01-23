import cv2
import time

# Bild einlesen
image = cv2.imread('testpicture_1.png')

#ab hier die Zeitmessung starten
startzeit = time.perf_counter_ns()
# In Graustufen umwandeln
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Bildrauschen reduzieren
blur_image = cv2.GaussianBlur(gray_image,(5,5),0)
# Canny-Kantenerkennung anwenden
canny_image = cv2.Canny(blur_image, 50, 150)
# Zeitmessung stoppen
endezeit = time.perf_counter_ns()
# Berechnete Zeit in ms ausgeben
print(f"Verarbeitungszeit: {(endezeit - startzeit)/1_000_000:.2f} ms")
# Angabe in fps (frames per second)
print(f"Entspricht ca. {1_000/((endezeit - startzeit)/1_000_000):.2f} fps")

# Ergebnis anzeigen
cv2.imshow("result", canny_image)
cv2.waitKey(0)
cv2.destroyAllWindows()