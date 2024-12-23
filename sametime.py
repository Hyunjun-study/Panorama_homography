import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def compute_canvas_size_and_offset(reference_img, img_list, homographies):
    ref_h, ref_w, _ = reference_img.shape
    corners = np.array([[0, 0], [ref_w, 0], [ref_w, ref_h], [0, ref_h]], dtype=np.float32).reshape(-1, 1, 2)

    all_corners = [corners]
    for img, H in zip(img_list, homographies):
        h, w = img.shape[:2]
        img_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        transformed_corners = cv.perspectiveTransform(img_corners, H)
        all_corners.append(transformed_corners)

    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    offset = (-x_min, -y_min)

    return canvas_width, canvas_height, offset

def compute_overlap_mask(canvas_shape, img, H, offset):
    h, w = img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv.perspectiveTransform(corners, H) + offset

    mask = np.zeros(canvas_shape[:2], dtype=np.uint8)
    cv.fillConvexPoly(mask, np.int32(transformed_corners), 255)
    return mask

def compute_bounding_box(img, H, offset):
    h, w = img.shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv.perspectiveTransform(corners, H) + offset
    return transformed_corners

def warp_image(canvas, img, H, offset):
    h, w = canvas.shape[:2]
    H_inv = np.linalg.inv(H)

    # 캔버스 좌표 생성
    y_coords, x_coords = np.indices((h, w))
    coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones_like(x_coords).ravel()])
    coords[0] -= offset[0]
    coords[1] -= offset[1]

    # 역 매핑
    mapped_coords = H_inv @ coords
    mapped_coords /= mapped_coords[2]

    mapped_x = mapped_coords[0].reshape(h, w).astype(np.int32)
    mapped_y = mapped_coords[1].reshape(h, w).astype(np.int32)

    # 유효 범위 확인
    valid_idx = (0 <= mapped_x) & (mapped_x < img.shape[1]) & (0 <= mapped_y) & (mapped_y < img.shape[0])
    canvas[valid_idx] = img[mapped_y[valid_idx], mapped_x[valid_idx]]

def blend_images(canvas, weight_mask):
    canvas_blend = np.zeros_like(canvas, dtype=np.float32)
    for i in range(3):
        canvas_blend[..., i] = canvas[..., i] * weight_mask
    return canvas_blend.astype(np.uint8)

def compute_overlap_region(mask1, mask2):
    return cv.bitwise_and(mask1, mask2)

def create_stronger_weighted_masks(overlap_mask, direction='horizontal', reverse=False, strength=4.0, blur_size=(15, 15)):
    h, w = overlap_mask.shape
    gradient = None

    if direction == 'horizontal':
        gradient = np.linspace(0, 1, w, dtype=np.float32).reshape(1, -1)
        if reverse:
            gradient = gradient[:, ::-1]  # Reverse gradient for right-to-left weight
        gradient = np.tile(gradient, (h, 1))
    elif direction == 'vertical':
        gradient = np.linspace(0, 1, h, dtype=np.float32).reshape(-1, 1)
        if reverse:
            gradient = gradient[::-1, :]
        gradient = np.tile(gradient, (1, w))

    gradient = np.power(gradient, strength)
    weighted_mask = (overlap_mask / 255.0) * gradient

    # 가우시안 블러 적용
    weighted_mask = cv.GaussianBlur(weighted_mask, blur_size, 0)
    return weighted_mask


def blend_images_with_masks(canvas, img, mask, H, offset, bbox):
    h, w = canvas.shape[:2]
    H_inv = np.linalg.inv(H)

    x_min, y_min, x_max, y_max = bbox

     # 캔버스 좌표 생성
    y_coords, x_coords = np.indices((y_max - y_min + 1, x_max - x_min + 1))
    coords = np.stack([x_coords.ravel() + x_min, y_coords.ravel() + y_min, np.ones_like(x_coords).ravel()])
    coords[0] -= offset[0]
    coords[1] -= offset[1]

    # 역 매핑
    mapped_coords = H_inv @ coords
    mapped_coords /= mapped_coords[2]

    mapped_x = mapped_coords[0].reshape(y_max - y_min + 1, x_max - x_min + 1).astype(np.int32)
    mapped_y = mapped_coords[1].reshape(y_max - y_min + 1, x_max - x_min + 1).astype(np.int32)

    # 유효 범위 확인
    valid_idx = (0 <= mapped_x) & (mapped_x < img.shape[1]) & (0 <= mapped_y) & (mapped_y < img.shape[0])
    valid_idx &= (strong_weighted_mask2_1 > 0)
    valid_idx &= (strong_weighted_mask3_2 > 0)


    # 블렌딩 처리
    for i in range(3):
        canvas[y_min:y_max + 1, x_min:x_max + 1, i][valid_idx] = (
            canvas[y_min:y_max + 1, x_min:x_max + 1, i][valid_idx] * (1 - mask[valid_idx]) +
            img[mapped_y[valid_idx], mapped_x[valid_idx], i] * mask[valid_idx]
        )

def resize_mask_to_canvas(mask, canvas_shape):
    return cv.resize(mask, (canvas_shape[1], canvas_shape[0]), interpolation=cv.INTER_LINEAR)

def apply_blended_mask(canvas, img, mask, H, offset):
    h, w = canvas.shape[:2]
    H_inv = np.linalg.inv(H)

    # 캔버스 좌표 생성
    y_coords, x_coords = np.indices((h, w))
    coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones_like(x_coords).ravel()])
    coords[0] -= offset[0]
    coords[1] -= offset[1]

    # 역 매핑
    mapped_coords = H_inv @ coords
    mapped_coords /= mapped_coords[2]

    mapped_x = mapped_coords[0].reshape(h, w).astype(np.int32)
    mapped_y = mapped_coords[1].reshape(h, w).astype(np.int32)

    # 유효 범위 확인
    valid_idx = (0 <= mapped_x) & (mapped_x < img.shape[1]) & (0 <= mapped_y) & (mapped_y < img.shape[0])

    # 마스크 적용하여 블렌딩
    for i in range(3):  # 채널별로 처리
        canvas[..., i][valid_idx] = (
            canvas[..., i][valid_idx] * (1 - mask[valid_idx]) +
            img[mapped_y[valid_idx], mapped_x[valid_idx], i] * mask[valid_idx]
        )

# 이미지 로드
img1 = cv.imread('nice_1.jpg')
img2 = cv.imread('nice_2.jpg')
img3 = cv.imread('nice_3.jpg')


# SIFT 키포인트 탐지 및 매칭
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
kp3, des3 = sift.detectAndCompute(img3, None)

# FLANN 매칭
flann = cv.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
matches1 = flann.knnMatch(des1, des2, k=2)
matches3 = flann.knnMatch(des3, des2, k=2)

good_matches1 = [m for m, n in matches1 if m.distance / n.distance < 0.7]
src_pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches1]).reshape(-1, 1, 2)
dst_pts1 = np.float32([kp2[m.trainIdx].pt for m in good_matches1]).reshape(-1, 1, 2)
H12 = cv.findHomography(src_pts1, dst_pts1, cv.RANSAC, 5.0)[0]

good_matches3 = [m for m, n in matches3 if m.distance / n.distance < 0.7]
src_pts3 = np.float32([kp3[m.queryIdx].pt for m in good_matches3]).reshape(-1, 1, 2)
dst_pts3 = np.float32([kp2[m.trainIdx].pt for m in good_matches3]).reshape(-1, 1, 2)
H32 = cv.findHomography(src_pts3, dst_pts3, cv.RANSAC, 5.0)[0]

# 동적으로 캔버스 크기와 오프셋 계산
canvas_w, canvas_h, offset = compute_canvas_size_and_offset(img2, [img1, img3], [H12, H32])

# 캔버스 초기화
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

warp_image(canvas, img2, np.eye(3), offset)
warp_image(canvas, img1, H12, offset)
warp_image(canvas, img3, H32, offset)


# 겹치는 마스크 계산
mask1 = compute_overlap_mask((canvas_h, canvas_w), img1, H12, offset)
mask2 = compute_overlap_mask((canvas_h, canvas_w), img2, np.eye(3), offset)
mask3 = compute_overlap_mask((canvas_h, canvas_w), img3, H32, offset)

# 겹치는 영역 계산
overlap_mask1_2 = compute_overlap_region(mask1, mask2)
overlap_mask2_3 = compute_overlap_region(mask2, mask3)


# 가중치 마스크 생성
strong_weighted_mask2_1 = create_stronger_weighted_masks(overlap_mask1_2, direction='horizontal', reverse=False, strength=10.0, blur_size=(11, 11))
strong_weighted_mask3_2 = create_stronger_weighted_masks(overlap_mask2_3, direction='horizontal', reverse=True, strength=50.0, blur_size=(31, 31))

# 마스크 정규화
strong_weighted_mask2_1 = strong_weighted_mask2_1 / np.max(strong_weighted_mask2_1)
strong_weighted_mask3_2 = strong_weighted_mask3_2 / np.max(strong_weighted_mask3_2)


# 가중치 마스크 캔버스 크기로 조정
strong_weighted_mask2_1 = resize_mask_to_canvas(strong_weighted_mask2_1, canvas.shape)
strong_weighted_mask3_2 = resize_mask_to_canvas(strong_weighted_mask3_2, canvas.shape)


# 모든 마스크를 이미지2에 적용
apply_blended_mask(canvas, img2, strong_weighted_mask2_1, np.eye(3), offset)
apply_blended_mask(canvas, img2, strong_weighted_mask3_2, np.eye(3), offset)



highlighted_canvas = canvas.copy()
highlighted_canvas[strong_weighted_mask2_1 > 0] = [0, 255, 0]  # 겹치는 영역 확인
highlighted_canvas[strong_weighted_mask3_2 > 0] = [255, 0, 0]  # 겹치는 영역 확인

plt.figure(figsize=(10, 5))
plt.imshow(cv.cvtColor(highlighted_canvas, cv.COLOR_BGR2RGB))
plt.title("Mask Region")
plt.axis('off')
plt.show()


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(strong_weighted_mask2_1, cmap='gray')
plt.title(" Mask 2->1")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(strong_weighted_mask3_2, cmap='gray')
plt.title(" Mask 3->2")
plt.axis('off')
plt.show()

# 최종 결과 시각화
plt.figure(figsize=(10, 5))
plt.imshow(cv.cvtColor(canvas, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("SameTime Final Image")
plt.show()
