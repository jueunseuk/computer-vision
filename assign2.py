import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==============================
# 1. 이미지 불러오기
# ==============================
image = cv2.imread("c:/junsu/image/food.jpg")

if image is None:
    raise FileNotFoundError("food.jpg 파일을 찾을 수 없습니다.")

# 컬러 → 그레이스케일
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ==============================
# 2. Sobel 미분
# ==============================
# x 방향 미분
dx = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

# y 방향 미분
dy = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)


# ==============================
# 3. (y, x) = (30, 40)
# ==============================
y = 30
x = 40

dx_value = dx[y, x]
dy_value = dy[y, x]

# 에지 강도
magnitude = np.sqrt(
    dx_value ** 2 +
    dy_value ** 2
)

# gradient 방향 (radian)
direction_rad = np.arctan2(
    dy_value,
    dx_value
)

# radian → degree
direction_deg = np.degrees(direction_rad)


print("===== (y, x) = (30, 40) =====")
print(f"dx = {dx_value}")
print(f"dy = {dy_value}")
print(f"에지 강도 = {magnitude}")
print(f"그레디언트 방향(rad) = {direction_rad}")
print(f"그레디언트 방향(degree) = {direction_deg}")


# ==============================
# 4. Sobel Edge Strength Map
# ==============================
sobel_magnitude = np.sqrt(
    dx ** 2 +
    dy ** 2
)

# 화면에 출력하기 위해 0~255 범위로 정규화
sobel_map = cv2.normalize(
    sobel_magnitude,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

sobel_map = sobel_map.astype(np.uint8)


# ==============================
# 5. Gaussian Smoothing
# ==============================
# sigmaX = sigmaY = 1
# 3x3 Gaussian filter
gaussian = cv2.GaussianBlur(
    gray,
    (3, 3),
    sigmaX=1,
    sigmaY=1
)


# ==============================
# 6. Canny Edge
# ==============================
# threshold가 문제에 지정되어 있지 않으므로
# 일반적인 예로 100, 200 사용
canny = cv2.Canny(
    gaussian,
    100,
    200
)


# ==============================
# 7. 결과 출력
# ==============================
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Original Grayscale")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(sobel_map, cmap="gray")
plt.title("Sobel Edge Strength Map")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(gaussian, cmap="gray")
plt.title("Gaussian Smoothing (3x3, sigma=1)")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(canny, cmap="gray")
plt.title("Canny Edge")
plt.axis("off")

plt.tight_layout()
plt.show()


# ==============================
# 8. 결과 이미지 저장
# ==============================
cv2.imwrite("sobel_edge.jpg", sobel_map)
cv2.imwrite("gaussian.jpg", gaussian)
cv2.imwrite("canny_edge.jpg", canny)