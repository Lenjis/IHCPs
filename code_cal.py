import json

def count_code_lines(ipynb_file):
    with open(ipynb_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    code_lines = 0

    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            code_lines += sum(1 for line in cell['source'] if line.strip())

    return code_lines

# 使用示例
ipynb_file = 'IHCP_flight_pm1000/20240607prototype.ipynb'  # 替换为你的 .ipynb 文件路径
code_lines = count_code_lines(ipynb_file)
print(f"Total lines of code: {code_lines}")
