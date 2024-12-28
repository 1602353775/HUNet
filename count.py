import os
import pandas as pd

def count_calligraphy_samples(root_dir):
    # Define font types
    font_types = ['ks', 'xs', 'cs', 'ls', 'zs']
    
    # Initialize list to store character statistics
    characters_stats = []

    # Traverse each subdirectory in the root directory
    for character in os.listdir(root_dir):
        char_path = os.path.join(root_dir, character)
        if os.path.isdir(char_path):
            # Initialize statistics for the current character
            char_data = {'汉字': character}
            total_count = 0  # Used to count the total samples for the current character
            
            # Count samples for each font type
            for font in font_types:
                count = len([file for file in os.listdir(char_path) if file.startswith(font)])
                char_data[font] = count
                total_count += count
                
            
            # Add total sample count for the character
            char_data['汉字样本量'] = total_count
            
            # Append statistics for the current character to the list
            characters_stats.append(char_data)
    
    # Convert the list to a DataFrame
    df = pd.DataFrame(characters_stats, columns=['汉字'] + font_types + ['汉字样本量'])
    
    # Add total sample count for each font type
    total_row = df.sum(numeric_only=True)
    total_row['汉字'] = '总计'
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    # Save the result to an Excel file in the root directory
    excel_path = os.path.join(root_dir, 'calligraphy_statistics.xlsx')
    df.to_excel(excel_path, index=False)
    
    return excel_path

# Example usage
# Replace 'test_1' with the actual path to your calligraphy dataset
excel_file_path = count_calligraphy_samples('dataset/test_1' )
print("Statistical results saved to:", excel_file_path)

import os

def print_images_without_prefix(root_dir):
    # 遍历根目录及其子目录
    for root, dirs, files in os.walk(root_dir):
        # 遍历当前目录下的所有文件
        for file in files:
            # 检查文件名是否以指定的前缀开头
            if not file.startswith(('ks', 'xs', 'cs', 'ls', 'zs')):
                # 如果不以指定前缀开头，则打印文件路径
                print(os.path.join(root, file))

# 调用函数并传入根目录路径
print_images_without_prefix('dataset/test_1' )

