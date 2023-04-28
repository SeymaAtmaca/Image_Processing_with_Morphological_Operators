import cv2
import matplotlib.pyplot as plt
import numpy as np

# ======= NOT : ARA GORUNTU CIKTILARININ GORULMESİ ICIN CV2.IMSHOW() FONKSIYONLARINI YORUM SATIRINDAN CIKARABILIRSINIZ =========

# goruntuyu okuma
cell = cv2.imread('./cell.jpg')
cells = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)
# cv2.imshow("cells", cells)

# ======================== orijinal goruntu uzerinde hucre cekirdegi bulma islemi ========================
# threshold, open ve close islemleri
_, thresholds = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
morphological_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
BW_opened = cv2.morphologyEx(thresholds, cv2.MORPH_OPEN, morphological_kernel)
# cv2.imshow("BW_opened", BW_opened)
BW_closed = cv2.morphologyEx(BW_opened, cv2.MORPH_CLOSE, morphological_kernel)
# cv2.imshow("BW_closed", BW_closed)


# baglantili bilesen analizi ile alani 80 < x < 700 olmayan hucreleri silme
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(BW_closed)
areas = stats[:, cv2.CC_STAT_AREA]
cell_area_max = 700
cell_area_min = 80
indices = np.where((areas < cell_area_max ) & (areas > cell_area_min))[0]



# ======================== Goruntunun tersi ile hucrelerin ceperleri icin islem yapilmasi ========================
# goruntunun tersi alinir.
cells_not = cv2.bitwise_not(cells)
# cv2.imshow("cells_not", cells_not)
# goruntu uzerinde hucrelerin daha iyi elde edilebilmesi icin threshold uygulanir.
_, thresholds_not = cv2.threshold(cells_not, 1,25, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow("thresholds_not", thresholds_not)
# elipse sekilli yapilandirma elemani olsuturulur.
kernel_not = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# hucre disi gurultunun azaltilabilmesi icin opening uygulanir.
BW_opened_not = cv2.morphologyEx(thresholds_not, cv2.MORPH_OPEN, kernel_not)
# cv2.imshow("BW_opened_not", BW_opened_not)
# hucre ici gurultunun giderilebilmesi icin closing uygulanir.
BW_closed_not = cv2.morphologyEx(BW_opened_not, cv2.MORPH_CLOSE, kernel_not)
# cv2.imshow("BW_closed_not", BW_closed_not)


# morfolojik islemler soncuu elde edilen goruntu uzerinde baglantili bilesen analizi yapilir.
_, labels_not, stats_not, centroids_not = cv2.connectedComponentsWithStats(BW_closed_not)
# bilesenlerin alanlari alinir.


# cekirdekli hucrelerin ayiklanabilmesi icin son bir maskeleme islemi yapilir.
# burada ilk elde edilen cekirdek koordinatlarinin, son elde edilen hucre koordinatlari arasinda kalip kalmadigi incelenir.
# eger cekirdek merkez degerleri hucrenin icinde kaliyorsa bu hucre cekirdekli oldugu icin alinir ve maskeye aktarilir.
masked_last = np.zeros_like(cells, dtype=np.uint8)
# çekirdek koordinatlarının hangi bileşenin içinde kaldığını bulma
count = 0
if len(indices) > 0: 
    for c in indices:
        # cekirdek merkez noktasi alinir.
        x, y = centroids[c].astype(int)[0], centroids[c].astype(int)[1]
        
        for i in range(1, labels_not.max() + 1):
            # hucrelere ait genislik ve yukseklik degerleri alinir.
            left, top, width, height, _ = stats_not[i]
            # eger incelenen cekirdek merkezi, hucre kordinati icindeyse ;
            if left <= x <= (left + width) and top <= y <= (top + height):
                # hucre bilesen indeksi ve cekirdek merkez kordinati basilir.
                count += 1
                print(f"Coordinate ({x}, {y}) is inside component {i}")
                # maskeye ceper degerleri islenir.
                masked_last[labels_not == i] = 1
                # maskelem ile, tersi alinan hucre uzerindeki cekirdekli hucre ceperleri alinir ve and_operation_last a atilir.
                and_operation_last = cv2.bitwise_and(cells_not, cells_not, mask = masked_last)
                # gurultu gidermek icin yapilandirma elemani olustuurlur ve closing islemi yapilir.
                close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
                nucleated_cells = cv2.morphologyEx( and_operation_last, cv2.MORPH_CLOSE, close)

print("Count of cells : ", count)
# cv2.imshow("Nucleated Cells",nucleated_cells)

# cekirdekli hucrelerde isarteleme yapabilmek icin erosion yapilir ve aradaki fark alinir.
erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
nucleated_cells_erosion = cv2.morphologyEx(nucleated_cells, cv2.MORPH_ERODE, erosion_kernel)
# cv2.imshow("Eroded nucleated cells", nucleated_cells_erosion)

# kenarlar elde edilir.
edges = nucleated_cells - nucleated_cells_erosion
# cv2.imshow("Edges", edges)


# olusturulan kenarlar rgb kanala tasinir ve isaretli sonuc goruntulenir.
cell[...,2] = cv2.bitwise_or(cell[... , 2] - edges, edges )
cv2.imshow("Marked cells", cell)


cv2.waitKey(0)
cv2.destroyAllWindows()