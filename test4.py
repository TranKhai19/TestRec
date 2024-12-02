import cv2
import numpy as np

# Đọc hình ảnh từ file
image_path = "test4.jpg"  # Thay bằng đường dẫn đúng
img = cv2.imread(image_path)

# Định nghĩa thông số DPI (dots per inch) và chuyển đổi pixel -> cm
DPI = 300  # Đổi theo thực tế của ảnh, ví dụ ảnh scan thường là 300 DPI
PIXEL_PER_CM = DPI / 2.54  # 1 inch = 2.54 cm

# Chuyển đổi sang ảnh xám và tìm vùng giấy trắng
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Tìm contour lớn nhất (giấy trắng) và tạo mask
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
mask = np.zeros_like(binary)
cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

# Tính diện tích giấy
paper_area_pixels = cv2.contourArea(largest_contour)
paper_area_cm2 = paper_area_pixels / (PIXEL_PER_CM**2)

# Áp dụng mask lên ảnh gốc để chỉ giữ tờ giấy trắng
result_img = cv2.bitwise_and(img, img, mask=mask)

# Tạo không gian màu HSV
hsv = cv2.cvtColor(result_img, cv2.COLOR_BGR2HSV)

# Định nghĩa các ngưỡng màu để phát hiện
color_ranges = {
    "Blue": [(90, 50, 50), (150, 255, 255)],
    "Red1": [(0, 100, 100), (10, 255, 255)],
    "Red2": [(160, 100, 100), (180, 255, 255)],
    "Yellow": [(20, 100, 100), (30, 255, 255)],
    "Green": [(40, 100, 100), (70, 255, 255)]
}

# Vẽ hình bao quanh các vùng màu sắc và tính thông số
output_image = result_img.copy()

# Duyệt qua các ngưỡng màu và tìm các vùng màu sắc
for color_name, (lower, upper) in color_ranges.items():
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    
    # Tạo mask cho màu cụ thể
    color_mask = cv2.inRange(hsv, lower, upper)
    color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)  # Áp dụng mask tờ giấy trắng

    # Tìm các contour trong mask màu sắc
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    color_area_pixels = 0

    for contour in contours:
        if cv2.contourArea(contour) > 200:  # Ngưỡng diện tích để lọc nhiễu
            # Tính toán các thông số
            x, y, w, h = cv2.boundingRect(contour)  # Hình chữ nhật bao quanh contour
            area_pixels = cv2.contourArea(contour)  # Diện tích của contour
            color_area_pixels += area_pixels

            # Chuyển đổi các thông số từ pixel sang cm
            width_cm = w / PIXEL_PER_CM
            height_cm = h / PIXEL_PER_CM
            area_cm2 = area_pixels / (PIXEL_PER_CM**2)
            percentage = (area_pixels / paper_area_pixels) * 100

            # Vẽ hình bao quanh và hiển thị thông tin
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                output_image,
                f"{color_name}: {percentage:.2f}%",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # In thông tin ra console
            print(f"Color: {color_name}")
            print(f"Width: {width_cm:.2f} cm, Height: {height_cm:.2f} cm")
            print(f"Area: {area_cm2:.2f} cm²")
            print(f"Percentage of Paper: {percentage:.2f}%\n")

# Hiển thị kết quả
cv2.imshow("Detected Colors on Paper", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
