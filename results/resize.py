import os
from PIL import Image


def resize_images(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith('.'):
                # 删除以'.'开头的隐藏文件
                file_path = os.path.join(root, file)
                os.remove(file_path)
                continue

            file_path = os.path.join(root, file)
            try:
                # 打开图片文件
                img = Image.open(file_path)

                # 确定缩放尺寸
                width, height = img.size
                if width >= height:
                    new_width = 128
                    new_height = int(height * (128 / width))
                else:
                    new_height = 128
                    new_width = int(width * (128 / height))

                # 缩放图片
                img = img.resize((new_width, new_height), Image.ANTIALIAS)

                # 保存缩放后的图片，覆盖原始文件
                img.save(file_path)

            except (IOError, OSError):
                # 如果打开图片失败，或者保存缩放后的图片失败，直接删除原始图片
                 print(file_path)
                # os.remove(file_path)


resize_images(r'D:\wangzhaojiang\书法真迹\ygsf_zj\以观书法\ks')