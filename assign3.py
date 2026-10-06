import cv2
import numpy as np
import os

# 영상 파일
input_path = "second_HW_lane_detection.mp4"
output_path = "lane_detection_result.mp4"

# 결과 프레임 저장 폴더
frame_dir = "result_frames"
os.makedirs(frame_dir, exist_ok=True)


# ROI 설정
def region_of_interest(image):
    height, width = image.shape[:2]

    mask = np.zeros_like(image)

    # 도로 영역
    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.42), int(height * 0.60)),
        (int(width * 0.58), int(height * 0.60)),
        (int(width * 0.95), height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(image, mask)


# 차선 그리기
def draw_lines(frame, lines):
    line_image = np.zeros_like(frame)

    if lines is None:
        return frame

    for line in lines:
        # 배열 형태와 관계없이 1차원으로 변환
        x1, y1, x2, y2 = line.reshape(-1)

        # 수직선 방지
        if x2 - x1 == 0:
            continue

        slope = (y2 - y1) / (x2 - x1)

        # 수평에 가까운 선 제거
        if abs(slope) < 0.5:
            continue

        cv2.line(
            line_image,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            5
        )

    # 원본과 차선 합성
    result = cv2.addWeighted(
        frame,
        0.8,
        line_image,
        1.0,
        0
    )

    return result


# 영상 불러오기
cap = cv2.VideoCapture(input_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 결과 영상 저장
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

# 서로 다른 4개 시점
save_frames = [
    int(frame_count * 0.2),
    int(frame_count * 0.4),
    int(frame_count * 0.6),
    int(frame_count * 0.8)
]

frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 회색조 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 노이즈 제거
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 에지 검출
    edges = cv2.Canny(blur, 50, 150)

    # 도로 영역만 선택
    roi = region_of_interest(edges)

    # 직선 검출
    lines = cv2.HoughLinesP(
        roi,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=100
    )

    # 차선 표시
    result = draw_lines(frame, lines)

    # 결과 영상 저장
    out.write(result)

    # 4개의 서로 다른 프레임 저장
    if frame_number in save_frames:
        index = save_frames.index(frame_number) + 1

        cv2.imwrite(
            f"{frame_dir}/frame_{index}.jpg",
            result
        )

    # 화면 출력
    cv2.imshow("Lane Detection", result)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_number += 1


cap.release()
out.release()
cv2.destroyAllWindows()

print("차선 검출 완료")
print("결과 영상:", output_path)
print("결과 프레임:", frame_dir)