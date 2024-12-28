import openpyxl
from shutil import copyfile
import os


def copy_images_from_excel(excel_file, output_folder):
    # 打开Excel文件
    wb = openpyxl.load_workbook(excel_file)
    sheet = wb.active
    # 遍历每一行数据
    for row in sheet.iter_rows(min_row=2, values_only=True):
        img_path = row[0]

        # 检查图片路径是否存在
        if not os.path.exists(img_path):
            print(f"图片路径不存在: {img_path}")
            continue

        img_label = str(row[1])
        pred_label = str(row[2])
        # 构建新的文件名
        filename = f"{img_label}-{pred_label}-{os.path.basename(img_path)}"
        # 复制图片到指定文件夹
        new_path = os.path.join(output_folder, filename)
        copyfile(img_path, new_path)
    # 关闭Excel文件
    wb.close()
# 调用函数，传入Excel文件路径和输出文件夹路径
excel_file = r"D:\wangzhaojiang\FLENet\experiments\Model_architecture\results\val_Error statistics\mobilenet_v3_large\19.xlsx"
output_folder = r"D:\wangzhaojiang\书法真迹\err_predict\mobilenetv3_large_19"
copy_images_from_excel(excel_file, output_folder)