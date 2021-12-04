import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np


def showimg4(img1, img2, img3, img4):
    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.title('Image original')
    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.title('Erode + dilate')
    plt.subplot(2, 2, 3)
    plt.imshow(img3, cmap='gray')
    plt.axis('off')
    plt.title('Image grey')
    plt.subplot(2, 2, 4)
    plt.imshow(img4, cmap='gray')
    plt.axis('off')
    plt.title('Otcu')
    plt.show()


image = cv2.imread("spoons_and_sugar.png")[:, :, ::-1]

# элиптическое ядро MORPH_ELLIPSE
hole_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

# Применяем эрозию и дилатацию, чтобы избавиться от белых линий
mask_image = cv2.erode(image, hole_mask)
mask_image = cv2.dilate(mask_image, hole_mask)

# Переводим изображение mask_image в полутоновое изображение
image_grey = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

# Применяем бинаризацию по Отсу к полутоновому изображению
_, bin_image_grey = cv2.threshold(image_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
showimg4(image, mask_image, image_grey, bin_image_grey)

# Находим контуры
contours, hierarchy = cv2.findContours(bin_image_grey, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Сортируем контуры
output_image = image.copy()
cnt_area = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    cnt_area.append(area)
indices = np.array(cnt_area).argsort()
sorted_area = sorted(cnt_area)
print(f"Отсортированные площади: {sorted_area}")

# Находим наибольшую разницу между областями
border_val_id = np.argmax(np.diff(sorted_area)) + 1
border_value = sorted_area[border_val_id]

# Разбиваем контуры на объекты - сахар и ложки
sugar = indices[:border_val_id]
spoons = indices[border_val_id:]

# Отрисовываем контуры по классам на изображение
# Выводим количественную информацию
countour_img = image.copy()

for i in indices:
    if i in sugar:
        cv2.drawContours(countour_img, contours, i, (0, 255, 0), 3)
    else:
        cv2.drawContours(countour_img, contours, i, (255, 0, 0), 3)

plt.imshow(countour_img, cmap='gray')
plt.axis('off')
plt.title(f'Final image: total objects = {len(indices)}, spoons = {len(spoons)}, sugar = {len(sugar)}')
plt.show()
