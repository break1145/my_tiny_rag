

import json

def file_to_list(filename: str) -> list:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        if not content.strip():
             raise json.JSONDecodeError("File is empty or contains only whitespace.", content, 0)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"文件 {filename} 内容不是有效的 JSON 格式: {e.msg}", e.doc, e.pos) from None

    return [data]