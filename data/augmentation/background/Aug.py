import cv2
import numpy as np

def erode_img(img, kernel_sizes):
    for i, kernel_size in enumerate(kernel_sizes):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        erosion = cv2.erode(img, kernel, iterations=1)
        cv2.imwrite(f'erode_{i}.jpg', erosion)

def dilate_img(img, kernel_sizes):
    for i, kernel_size in enumerate(kernel_sizes):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilation = cv2.dilate(img, kernel, iterations=1)
        cv2.imwrite(f'dilate_{i}.jpg', dilation)


def rotate_img(img, num_rotations):
    for i in range(num_rotations):
        angle = np.random.uniform(-30, 30)  # 生成-30到30之间的随机数
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

        # 如果图像是灰度图像
        if len(img.shape) == 2:
            borderValue = 255
        else:  # 彩色图像
            borderValue = (255, 255, 255)

        dst = cv2.warpAffine(img, M, (cols, rows), borderValue=borderValue)

        cv2.imwrite(f'rotate_{i}.jpg', dst)

def perspective_transform(img, pts1, pts2,rows, cols):
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, matrix, (cols, rows))
    cv2.imwrite('transform.jpg', dst)


def add_noise(img, noise_levels):
    for i, noise_level in enumerate(noise_levels):
        noise = np.random.normal(0, noise_level, img.shape)
        img_noise = img + noise
        cv2.imwrite(f'noise_{i}.jpg', img_noise)

def main():

    # 读取图片
    img = cv2.imread('/Volumes/wangzhaojiang/FLENet/data/augmentation/background/23.png', 0)

    # 数据增强
    erode_img(img, [9, 5, 7])
    dilate_img(img, [9, 5, 7])
    rotate_img(img, 10)  # 生成10个-30到30度旋转的图片
    rows, cols = img.shape
    pts1 = np.float32([[50,50],[200,50],[50,200],[200,200]])
    pts2 = np.float32([[10,100],[200,50],[100,250],[200,250]])

    # 数据增强
    perspective_transform(img, pts1, pts2,rows, cols)

    add_noise(img, [0.1, 0.5, 1.0])

if __name__ == "__main__":
    main()
