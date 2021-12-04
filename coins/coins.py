from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import imutils
import cv2.cv2 as cv2
import matplotlib.pyplot as plt


def showimg8(img1, img2, img3, img4, img5, img6, img7, img8):
    plt.subplot(2, 4, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.title('Original')
    plt.subplot(2, 4, 2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.title('Laplacian')
    plt.subplot(2, 4, 3)
    plt.imshow(img3, cmap='gray')
    plt.axis('off')
    plt.title('Original + Laplacian')
    plt.subplot(2, 4, 4)
    plt.imshow(img4, cmap='gray')
    plt.axis('off')
    plt.title('Sobel')
    plt.subplot(2, 4, 5)
    plt.imshow(img5, cmap='gray')
    plt.axis('off')
    plt.title('Smoothing, 5x5')
    plt.subplot(2, 4, 6)
    plt.imshow(img6, cmap='gray')
    plt.axis('off')
    plt.title('Mask=(Original+Laplacian)*Smoothing')
    plt.subplot(2, 4, 7)
    plt.imshow(img7, cmap='gray')
    plt.axis('off')
    plt.title('Original + Mask')
    plt.subplot(2, 4, 8)
    plt.imshow(img8, cmap='gray')
    plt.axis('off')
    plt.title('Gamma correction')
    plt.show()


def showimg4(img1, img2, img3, img4, title=''):
    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.title('Original')
    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.title('Filtered image')
    plt.subplot(2, 2, 3)
    plt.imshow(img3, cmap='gray')
    plt.axis('off')
    plt.title('Otsu')
    plt.subplot(2, 2, 4)
    plt.imshow(img4, cmap='gray')
    plt.axis('off')
    plt.title('Final image: ' + title)
    plt.show()


def normalize_image(np_img, mul_const=1.0):
    return (mul_const * (np_img - np_img.min()) / (np_img.max() - np_img.min()))


def cut_img_values(img):
    img_copy = img
    img_copy[img_copy < 0] = 0
    img_copy[img_copy > 255] = 255
    return img_copy


def image_transforms(img):
    # (a) Исходное изображение
    # (б) Применение оператора лапласиана к изображению
    laplassian_img_cut = cut_img_values(cv2.Laplacian(255 - img, ddepth=cv2.CV_16U, ksize=3))
    laplassian_img = normalize_image(cv2.Laplacian(img, ddepth=cv2.CV_16U, ksize=3), 255.0).astype("int")
    laplassian_img = (255 * np.power(laplassian_img / 255, 0.5)).astype("uint8")

    # (в) Повышение резкости сложением изображений (а) и (б)
    laplassian_img_add = normalize_image(img + laplassian_img_cut, 255.0).astype("uint8")

    # (г) Применение градиентного оператора Собела к изображению (а)
    sobel_x = cv2.Sobel(img, ddepth=cv2.CV_16U, dx=1, dy=0)
    sobel_y = cv2.Sobel(img, ddepth=cv2.CV_16U, dx=0, dy=1)
    sobel = np.absolute(sobel_x) + np.absolute(sobel_y)
    sobel_img = cut_img_values(img + sobel)

    # (д) Изображение (г), сглаженное усредняющим фильтром по окрестности 5×5.
    sobel_blur = cv2.blur(sobel_img, (8, 8))

    # (е) Изображение-маска, полученное перемножением изображений (в) и (д)
    laplassian_blur_sobel_mask = normalize_image(
        cv2.multiply(laplassian_img_add, sobel_blur, dtype=cv2.CV_16U), 255.0).astype(int)

    # (ж) Изображение с повышенной резкостью, полученное сложением изображений (а) и (е).
    orig_laplassian_blur_sobel_mask = cut_img_values(img + laplassian_blur_sobel_mask)

    # (з) Конечный результат, полученный из изображения (ж) градационной коррекцией по степенному закону
    final_res = (255 * np.power(orig_laplassian_blur_sobel_mask / 255, 2.5)).astype("uint8")

    showimg8(img, laplassian_img, laplassian_img_add, sobel_img, sobel_blur,
             laplassian_blur_sobel_mask, orig_laplassian_blur_sobel_mask, final_res)

    return (img, laplassian_img, laplassian_img_add, sobel_img, sobel_blur,
            laplassian_blur_sobel_mask, orig_laplassian_blur_sobel_mask, final_res)


filename = "coins.jpg"
image = cv2.imread(filename)

# фильтрованние изображения с цветовыми градиентами и сглаженной мелкозернистой текстурой
shifted = (255 * np.power(cv2.pyrMeanShiftFiltering(image, 21, 51) / 255, 1)).astype("uint8")

# Яркостные преобразования и пространственная фильтрация - функция из задания про скелеты
transformed = image_transforms(shifted)

# Преобразование изображения к серому цвету
gray = cv2.cvtColor(transformed[7], cv2.COLOR_RGB2GRAY)

# Преобразование изображения к бинарному изображению
if (gray < 100).sum() < gray.size / 2:
    BINARY = cv2.THRESH_BINARY_INV
else:
    BINARY = cv2.THRESH_BINARY
thresh = cv2.threshold(gray, 60, 255, BINARY + cv2.THRESH_OTSU)[1]

# Нахождение градиента
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# вычисляем точное евклидово расстояние от каждого двоичного пикселя
# до ближайшего нулевого пикселя, затем найдем пики в этой карта расстояний
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20, labels=thresh)

# выполняем анализ связанных компонентов на локальных пиках,
# используя 8-связность, затем применяем алгоритм Watershed
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)

# проходим по уникальным этикеткам, возвращаемым алгоритм водораздела
radius_list = []
coords_list = []
for label in np.unique(labels):
    # если метка равна нулю, мы исследуем фон => игнорируем
    if label == 0:
        continue
    # в противном случае выделяем память для области метки и нарисуем это на маске
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    # обнаруживаем контуры в маске и берем самые крупные
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # рисуем круг, охватывающий объект
    ((x, y), r) = cv2.minEnclosingCircle(c)
    if r > 15:
        radius_list.append(r)
        coords_list.append((x, y))

radius_list = np.array(radius_list)
coords_list = np.array(coords_list)

# сортируем список радиусов, список координат
indices = radius_list.argsort()
radius_list = radius_list[indices[::1]]
coords_list = coords_list[indices[::1]]
print(radius_list)

border_val_id = np.argmax(np.diff(radius_list)) + 1
output_image = image.copy()
color = (0, 255, 0)

for i, r in enumerate(radius_list):
    (x, y) = coords_list[i]
    if i == border_val_id:
        color = (255, 0, 0)
    cv2.circle(output_image, (int(x), int(y)), int(r), color, 2)
    # cv2.putText(output_image, f"#{i + 1}", (int(x) - 10, int(y)),
    #            cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 0)

output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
output_title = f"total objects ={len(radius_list)}, small={border_val_id}, big={len(radius_list) - border_val_id}"
showimg4(image, transformed[7], thresh, output_image, output_title)
