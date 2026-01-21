import time
import math
import numpy as np
import cv2 as cv

# Cấu hình hình ảnh
M = 1024  # chiều rộng
N = 768   # chiều cao
CENTER = (M // 2, N // 2)  # tâm ảnh (x, y)
RADIUS = min(M, N) // 2 - 40  # bán kính mặt đồng hồ, để lại lề
LEVEL = 5  # mức (1..5). Nhấn phím 1-5 để đổi mức khi chạy.

# Màu sắc (BGR)
PURPLE = (128, 0, 128)  # tím nền
WHITE = (255, 255, 255)  # trắng
HOUR_COLOR = (255, 0, 0)   # kim giờ: xanh dương (BGR)
MIN_COLOR = (0, 255, 0)    # kim phút: xanh lá (BGR)
SEC_COLOR = (0, 0, 255)    # kim giây: đỏ (BGR)
NUM_COLORS = [  # Các màu luân phiên cho chữ La Mã
    (255, 255, 255),
    (200, 200, 50),
    (50, 200, 200),
    (200, 50, 200),
    (50, 255, 100),
    (255, 120, 50),
]

# La Mã cho các số giờ
ROMAN = ["XII", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI"]

FONT = cv.FONT_HERSHEY_SIMPLEX

def polar_to_cart(center, angle_deg, length):
    """
    Chuyển tọa độ cực (góc theo độ, độ dài) về tọa độ ảnh (x, y).

    Lưu ý: angle_deg ở đây được xử lý sao cho 0 độ là vị trí 12 giờ và góc tăng theo chiều kim đồng hồ.
    """
    angle = math.radians(angle_deg - 90)
    x = int(center[0] + length * math.cos(angle))
    y = int(center[1] + length * math.sin(angle))
    return (x, y)


def draw_clock_face(img):
    """
    Vẽ mặt đồng hồ: nền tròn tím, viền ngoài, các vạch chỉ phút/giờ và chữ La Mã.

    Việc hiển thị vạch phút đầy đủ chỉ xuất hiện khi LEVEL >= 5.
    """
    # Nền mặt đồng hồ
    cv.circle(img, CENTER, RADIUS, PURPLE, -1, lineType=cv.LINE_AA)

    # Viền ngoài
    cv.circle(img, CENTER, RADIUS, (80, 0, 80), 8, lineType=cv.LINE_AA)

    # Vạch phút (60 vạch). Vạch giờ (mỗi 5 phút) dài và dày hơn
    for i in range(60):
        angle_deg = i * 6  # 360/60
        inner = RADIUS - 15 if i % 5 == 0 else RADIUS - 8
        p1 = polar_to_cart(CENTER, angle_deg, inner)
        p2 = polar_to_cart(CENTER, angle_deg, RADIUS - 2)
        thickness = 3 if i % 5 == 0 else 1
        color = WHITE if i % 5 == 0 else (180, 180, 180)
        if LEVEL >= 5:
            # Nếu LEVEL >=5, vẽ tất cả 60 vạch phút
            cv.line(img, p1, p2, color, thickness, lineType=cv.LINE_AA)
        else:
            # Ngược lại chỉ vẽ vạch giờ (mỗi 5 phút)
            if i % 5 == 0:
                cv.line(img, p1, p2, color, thickness, lineType=cv.LINE_AA)

    # Vẽ chữ La Mã ở vị trí các giờ, màu luân phiên
    for i, label in enumerate(ROMAN):
        angle_deg = i * 30  # 360/12
        text_pos = polar_to_cart(CENTER, angle_deg, RADIUS - 50)
        text_size, _ = cv.getTextSize(label, FONT, 1.0, 2)
        text_x = text_pos[0] - text_size[0] // 2
        text_y = text_pos[1] + text_size[1] // 2
        color = NUM_COLORS[i % len(NUM_COLORS)]
        cv.putText(img, label, (text_x, text_y), FONT, 1.0, color, 2, lineType=cv.LINE_AA)


def draw_hand(img, angle_deg, length_frac, color, thickness):
    """
    Vẽ một kim đồng hồ từ tâm đến điểm cuối xác định bởi góc và tỉ lệ bán kính.

    length_frac: tỉ lệ so với RADIUS (ví dụ 0.9 = 90% bán kính).
    """
    end = polar_to_cart(CENTER, angle_deg, int(RADIUS * length_frac))
    cv.line(img, CENTER, end, color, thickness, lineType=cv.LINE_AA)


def draw_hub(img):
    """Vẽ tâm đồng hồ (hub) - tách ra để vẽ sau cùng lên trên kim."""
    cv.circle(img, CENTER, 8, (50, 50, 50), -1, lineType=cv.LINE_AA)
    cv.circle(img, CENTER, 4, (200, 200, 200), 1, lineType=cv.LINE_AA)


def get_time_components():
    """
    Lấy thời gian hiện tại và trả về:

    - hours: dạng số thực 0..11 (bao gồm phần phút để kim giờ di chuyển mượt)
    - minutes: dạng số thực 0..59 (bao gồm phần giây)
    - seconds: dạng số thực 0..59 (có phần thập phân)
    """
    t = time.time()
    local = time.localtime(t)
    frac = t - int(t)
    seconds = local.tm_sec + frac
    minutes = local.tm_min + seconds / 60.0
    hours = (local.tm_hour % 12) + minutes / 60.0
    return hours, minutes, seconds


def draw_hands_on(img, level):
    """
    Vẽ 3 kim: giây, phút, giờ. Hành vi tùy theo LEVEL.

    Hàm này chỉ vẽ kim trên ảnh đã được sao chép từ background.
    """
    hours, minutes, seconds = get_time_components()

    # Kim giây (tính góc từ giây)
    sec_angle = (seconds / 60.0) * 360.0
    if level >= 2:
        draw_hand(img, sec_angle, 0.9, SEC_COLOR, 1)

    # Kim phút (mượt nếu LEVEL >=3)
    min_angle = (minutes / 60.0) * 360.0
    if level >= 3:
        draw_hand(img, min_angle, 0.75, MIN_COLOR, 4)
    else:
        draw_hand(img, (int(minutes) / 60.0) * 360.0, 0.75, MIN_COLOR, 4)

    # Kim giờ (mượt nếu LEVEL >=4)
    hour_angle = (hours / 12.0) * 360.0
    if level >= 4:
        draw_hand(img, hour_angle, 0.5, HOUR_COLOR, 6)
    else:
        draw_hand(img, (int(hours) / 12.0) * 360.0, 0.5, HOUR_COLOR, 6)


def main():
    global LEVEL

    # Tạo cửa sổ hiển thị (không tạo lại mỗi frame)
    cv.namedWindow("Analog Clock", cv.WINDOW_AUTOSIZE)

    # --- Chống flicker: vẽ mặt đồng hồ tĩnh một lần vào ảnh nền (background),
    # sau đó mỗi frame chỉ sao chép background và vẽ kim lên bản sao đó. ---
    background = np.zeros((N, M, 3), dtype=np.uint8)
    draw_clock_face(background)  # vẽ mặt đồng hồ chỉ 1 lần

    while True:
        # Sao chép background để có frame mới, tránh vẽ lại toàn bộ mặt đồng hồ mỗi frame
        img = background.copy()

        # Vẽ kim lên ảnh sao chép
        draw_hands_on(img, LEVEL)

        # Vẽ hub sau cùng để đè lên phần giao nhau của kim (trông tự nhiên hơn)
        draw_hub(img)

        # Ghi chú UI: hiển thị mức hiện tại và hướng dẫn phím
        cv.putText(img, f"Level: {LEVEL}  (Press 1-5 to change, Esc to quit)", (20, 30),
                   FONT, 0.7, WHITE, 2, lineType=cv.LINE_AA)

        cv.imshow("Analog Clock", img)
        key = cv.waitKey(30) & 0xFF  # 30 ms cho ~33 FPS, đủ mượt cho kim giây
        if key == 27:  # ESC để thoát
            break
        if key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            LEVEL = int(chr(key))
            # Nếu LEVEL thay đổi có ảnh hưởng đến mặt đồng hồ (ví dụ LEVEL <5 làm ẩn vạch phút),
            # ta cần cập nhật lại background tương ứng
            background = np.zeros((N, M, 3), dtype=np.uint8)
            draw_clock_face(background)

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()