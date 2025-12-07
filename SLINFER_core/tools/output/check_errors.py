import os

# 获取当前目录
current_directory = os.getcwd()

# 遍历当前目录中的所有文件
for filename in os.listdir(current_directory):
    # 检查文件是否是 .out 文件
    if filename.endswith('.out'):
        # 打开文件并检查是否包含 "error" 字符串
        with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            if 'error' in content.lower():  # 使用 .lower() 使检查不区分大小写
                print(f'File "{filename}" contains "error".')