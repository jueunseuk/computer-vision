import cv2
import numpy as np

# 영상 읽기
src = cv2.imread('c:\junsu\image\BALLOON.bmp')

# BGR 영상을 HSV 컬러 공간으로 변환
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

# 파란색 HSV 범위 지정
lower_blue = np.array([100, 80, 50], dtype=np.uint8)
upper_blue = np.array([130, 255, 255], dtype=np.uint8)

# 파란색 영역만 마스크 생성
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 입력 영상과 같은 크기의 백색 배경 생성
dst = np.full(src.shape, 255, dtype=np.uint8)

# 마스크 영역의 파란 풍선만 백색 배경으로 복사
cv2.copyTo(src, mask, dst)

# 결과 출력
cv2.imshow("Original", src)
cv2.imshow("Blue Balloon", dst)

cv2.waitKey(0)
cv2.destroyAllWindows()