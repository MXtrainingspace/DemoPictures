import cv2
import numpy as np

# Funktion zur Berechnung der zirkulären Winkeldifferenz
def angle_difference(theta1, theta2):
    """Berechnet die minimale Winkeldifferenz (zirkulär)"""
    diff = abs(np.degrees(theta1 - theta2))
    if diff > 90:  # Winkel sind zirkulär (0-180)
        diff = 180 - diff
    return diff

# Funktion zum Bilden von Gruppen ähnlicher Linien
def group_lines_by_angle(lines, angle_threshold=3):
    """
    Bildet Gruppen von Linien, die einen ähnlichen Winkel (±angle_threshold Grad) haben.

    Args:
        lines: Hough-Linien von cv2.HoughLinesWithAccumulator (shape: [n, 1, 3])
        angle_threshold: Winkelschwelle in Grad für die Gruppierung (Standard: 3 Grad)


    Returns:
        Liste von Gruppen, jede Gruppe ist wiederum eine Liste von (rho, theta, votes) Tupeln
    """
    if lines is None or len(lines) < 1:
        return []

    # Linien in Liste konvertieren
    lines_list = [(rho, theta, votes) for rho, theta, votes in lines[:, 0]]
    lines_list.sort(key=lambda x: x[1])



    # Gruppen erstellen
    gruppen = []
    aktuelle_gruppe = [lines_list[0]]

    for i in range(1, len(lines_list)):
        rho, theta, votes = lines_list[i]
        prev_rho, prev_theta, prev_votes = aktuelle_gruppe[-1]
        # Winkeldifferenz berechnen
        angle_diff = angle_difference(theta, prev_theta)

        # Wenn Winkel ähnlich, zum aktuellen Cluster hinzufügen
        if angle_diff <= angle_threshold:
            aktuelle_gruppe.append((rho, theta, votes))
        else:
            # Cluster speichern und neuen starten
            gruppen.append(aktuelle_gruppe)
            aktuelle_gruppe = [(rho, theta, votes)]

    # Letzten Cluster nicht vergessen
    if aktuelle_gruppe:
        gruppen.append(aktuelle_gruppe)

    return gruppen

# Funktion zur Berechnung des Lenkwinkels
def get_line_x_at_y(rho, theta, y):
    """
    Berechnet die x-Position einer Hough-Linie bei gegebenem y

    Args:
        rho: Parameter der Hough-Linie
        theta: Winkel der Hough-Linie in Radiant
        y: y-Koordinate

    Returns:
        x-Koordinate oder None wenn nicht berechenbar
    """
    cos_theta = np.cos(theta)
    if abs(cos_theta) < 1e-6:  # Vermeiden von Division durch 0
        return None
    x = (rho - y * np.sin(theta)) / cos_theta
    return x

def calculate_steering_angle(top_two_clusters):
    """
    Berechnet den Lenkwinkel basierend auf zwei erkannten Fahrspuren
    Nutzt die globalen Variablen width und height

    Args:
        top_two_clusters: Liste der erkannten Spurcluster (rho, theta, count, votes)

    Returns:
        float: steering_angle_degrees (Lenkwinkel in Grad, links +, rechts -)
               oder None wenn nicht berechenbar
    """
    if len(top_two_clusters) < 2:
        return None

    rho1, theta1, count1, votes1 = top_two_clusters[0]
    rho2, theta2, count2, votes2 = top_two_clusters[1]

    y_target = height // 9  # Höhe, auf der die Spurmitte bestimmt wird (von oben gezählt)
    x1 = get_line_x_at_y(rho1, theta1, y_target)
    x2 = get_line_x_at_y(rho2, theta2, y_target)

    if x1 is not None and x2 is not None:
        lane_center = (x1 + x2) / 2

        # Lenkwinkel aus Linie vom unteren Bildzentrum zum Spurmittelpunkt auf y_target
        cx, cy = width / 2, height - 1
        dx = lane_center - cx
        dy = cy - y_target
        steering_angle_degrees = np.degrees(np.arctan2(dx, dy))

        return steering_angle_degrees

    return None

def sum_group_votes(group):
    """
    Berechnet die Summe der Votes (Bildpunkte) einer Gruppe

    Args:
        group: Liste von (rho, theta, votes) Tupeln

    Returns:
        int: Summe aller Votes in der Gruppe
    """
    return sum(v for r, t, v in group)

# Beginn Hauptprogramm
if __name__ == "__main__":
    # Bild laden
    komplettes_Bild = cv2.imread('testpicture_1.png')

    # Aufteilen in obere und untere Hälfte
    obere_haelfte = komplettes_Bild[: komplettes_Bild.shape[0] // 2, :]
    untere_haelfte = komplettes_Bild[komplettes_Bild.shape[0] // 2 :, :]
    # Größenanpassung um im folgenden Rechenzeit zu sparen
    ziel_groesse = (640, 240) # tupel für Breite, Höhe
    obere_haelfte = cv2.resize(obere_haelfte, ziel_groesse)
    untere_haelfte = cv2.resize(untere_haelfte, ziel_groesse)  # untere Hälfte wird im folgenden analysiert
    height, width = untere_haelfte.shape[:2]  # Höhe und Breite des (resized) unteren Bildes -> kann auch entfallen da oben in dem objekt resize_dim definiert
    threshold = int(np.sqrt(width**2 + height**2) * 0.05) # Hough-Threshold basierend auf 5% der Bilddiagonale
    # Alternativ: threshold = int(np.sqrt(resize_dim[0]**2 + resize_dim[1]**2) * 0.05)  oder
    # threshold = int(np.hypot(resize_dim[0], resize_dim[1]) * 0.05)

    untere_haelfte_kopie = untere_haelfte.copy()  # Kopie für die Visualisierung (Hough Transformation)

    # In Graustufen umwandeln
    graubild = cv2.cvtColor(untere_haelfte, cv2.COLOR_BGR2GRAY)

    # Kontrast erhöhen mit CLAHE (Contrast Limited Adaptive Histogram Equalization), Umfang ist nicht im Skript
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_erhoeht = clahe.apply(graubild)

    # Bildrauschen reduzieren, hier mit einem 15x15 Element
    gefiltertes_bild = cv2.GaussianBlur(contrast_erhoeht,(15,15),0)
    canny_edges = cv2.Canny(gefiltertes_bild, 50, 150)

    # Standard Hough Transformation in der Variante, dass die Anzahl Bildpunkte pro Linie zurückgegeben wird (Accumulator)
    lines = cv2.HoughLinesWithAccumulator(canny_edges, rho=1, theta=np.pi/180, threshold=threshold)


    # Finde die zwei häufigsten Cluster
    groups = group_lines_by_angle(lines, angle_threshold=10)
    groups.sort(key=sum_group_votes, reverse=True)

    # Durchschnittliche Linien (Durchschnitt für Theta und Rho in den beiden Gruppen mit den meisten votes/Bildpunkten)
    hauptlinien = []
    if len(groups) >= 2:
        for cluster in groups[:2]:
            avg_rho = np.mean([r for r, t, v in cluster])
            avg_theta = np.mean([t for r, t, v in cluster])
            total_votes = np.sum([v for r, t, v in cluster])
            count = len(cluster)
            hauptlinien.append((avg_rho, avg_theta, count, total_votes))
    elif len(groups) == 1:
        cluster = groups[0]
        avg_rho = np.mean([r for r, t, v in cluster])
        avg_theta = np.mean([t for r, t, v in cluster])
        total_votes = np.sum([v for r, t, v in cluster])
        count = len(cluster)
        hauptlinien.append((avg_rho, avg_theta, count, total_votes))



    # Berechne Lenkwinkel, wenn zwei Spuren erkannt wurden (Strategie: Mittelpunkt auf Höhe height/3)
    steering_angle_degrees = calculate_steering_angle(hauptlinien)

    # Berechnung der maximalen Linienlänge für die Visualisierung
    t = int(np.sqrt(untere_haelfte.shape[0]**2 + untere_haelfte.shape[1]**2))

    # Draw detected lines on the image (Standard Hough)
    if lines is not None:
        for rho, theta, votes in lines[:,0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + t * (-b))
            y1 = int(y0 + t * (a))
            x2 = int(x0 - t * (-b))
            y2 = int(y0 - t * (a))
            cv2.line(untere_haelfte_kopie, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw the two most frequent clusters as combined lines
    untere_haelfte_hauptgruppen = untere_haelfte.copy()
    colors = [(255, 0, 0), (0, 0, 255)]  # Blau und Rot
    for idx, (rho, theta, count, votes_sum) in enumerate(hauptlinien):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + t * (-b))
        y1 = int(y0 + t * (a))
        x2 = int(x0 - t * (-b))
        y2 = int(y0 - t * (a))
        cv2.line(untere_haelfte_hauptgruppen, (x1, y1), (x2, y2), colors[idx], 3)
        # Text an fester Position oben links platzieren
        text_y = 30 + idx * 35  # Abstand zwischen den Texten
        cv2.putText(obere_haelfte, f"Hauptlinie {idx+1} (Anzahl Linien {count} | summierte Bildpunkte {votes_sum:.0f})", (10, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], 2)

    # Zeichne Lenkwinkel-Linie (Start unten mittig, Ende direkt am Spurmittelpunkt)
    if steering_angle_degrees is not None and len(hauptlinien) >= 2:
        rho1, theta1, count1, votes1 = hauptlinien[0]
        rho2, theta2, count2, votes2 = hauptlinien[1]

        y_target = height // 9
        x1 = get_line_x_at_y(rho1, theta1, y_target)
        x2 = get_line_x_at_y(rho2, theta2, y_target)

        if x1 is not None and x2 is not None:
            lane_center = (x1 + x2) / 2
            cx = width // 2
            cy = height - 1
            end_point = (int(lane_center), int(y_target))
            cv2.arrowedLine(untere_haelfte_hauptgruppen, (cx, cy), end_point, (0, 255, 255), 3, tipLength=0.07)
            # Zielpunkt des Spurmittelpunkts auf y_target zur Kontrolle
            cv2.circle(untere_haelfte_hauptgruppen, (int(lane_center), y_target), 6, (0, 255, 255), -1)


    # Anzeigen und Ergebnisse ausgeben
    cv2.imshow('1. Graustufenbild', graubild)
    cv2.imshow('2. Kontrastverstaerktes Bild', contrast_erhoeht)
    cv2.imshow('3. Gefiltertes Bild', gefiltertes_bild)
    cv2.imshow('4. Canny Kantenerkennung', canny_edges)
    cv2.imshow('5. Standard Hough - alle Linien (rot)', untere_haelfte_kopie)
    # Oberhälfte wieder anfügen für die finale Ansicht
    ergebnisbild = np.vstack((obere_haelfte, untere_haelfte_hauptgruppen))
    cv2.imshow('6. Die zwei haeufigsten Cluster (blau und rot)', ergebnisbild)


    print(f"Erkannte Linien (gesamt): {len(lines) if lines is not None else 0}")
    print(f"Anzahl Cluster: {len(groups)}")
    if len(hauptlinien) >= 1:
        rho1, theta1, count1, votes1 = hauptlinien[0]
        print(f"Lane 1: {count1} Linien (rho={rho1:.2f}, theta={np.degrees(theta1):.2f}°, votes={votes1:.0f})")
    if len(hauptlinien) >= 2:
        rho2, theta2, count2, votes2 = hauptlinien[1]
        print(f"Lane 2: {count2} Linien (rho={rho2:.2f}, theta={np.degrees(theta2):.2f}°, votes={votes2:.0f})")

    # Gebe Lenkwinkel aus
    if steering_angle_degrees is not None:
        print(f"\nLenkwinkel:")
        print(f"  Lenkwinkel (Grad, links + / rechts -): {steering_angle_degrees:.2f}°")
        if steering_angle_degrees > 0:
            print(f"  Richtung: Lenken Sie nach LINKS")
        elif steering_angle_degrees < 0:
            print(f"  Richtung: Lenken Sie nach RECHTS")
        else:
            print(f"  Richtung: Geradeaus")
    else:
        print("\nLenkwinkel: Nicht berechenbar (weniger als 2 Spuren erkannt)")

    print("Bilder angezeigt. Beliebige Taste drücken (Mauszeiger muss über einem der angezeigten Bilder sein) zum Beenden.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()