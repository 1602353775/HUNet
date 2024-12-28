import os
def delete_hidden_files(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.startswith('.') and os.path.isfile(file_path):
                os.remove(file_path)

# folder_path = r"I:\Oracle-50K-125"
# delete_hidden_files(folder_path)


import os

def delete_empty_folders(path):
    if not os.path.isdir(path):  # 如果路径不是文件夹，则直接返回
        return

    # 获取当前文件夹的所有子文件夹
    subfolders = [os.path.join(path, subfolder) for subfolder in os.listdir(path)]

    for subfolder in subfolders:
        if os.path.isdir(subfolder):  # 如果子文件夹是文件夹
            delete_empty_folders(subfolder)  # 递归调用删除子文件夹中的空文件夹

    # 删除当前文件夹，如果该文件夹为空
    if not os.listdir(path):
        os.rmdir(path)


path = r"D:\wangzhaojiang\书法真迹\ygsf_zj\以观书法"
delete_empty_folders(path)