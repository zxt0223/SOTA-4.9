import cv2
import json
import numpy as np
import os

# 严格使用你的 A-ORENet 专属路径
JSON_DIR = '/group/chenjinming/Datas/test-img-json' # 你的原始标注路径 (请确认是否需要拷贝到 raw_json)
SAVE_DIR = 'A-ORENet_Datas/edge_1px_labels'

def generate_edge_labels():
    os.makedirs(SAVE_DIR, exist_ok=True)
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    
    if len(json_files) == 0:
        print(f"❌ 警告：在 {JSON_DIR} 下没有找到 JSON 文件！")
        return

    print(f"🔍 找到 {len(json_files)} 个标注文件，开始提取 1 像素物理边缘...")
    
    for json_file in json_files:
        with open(os.path.join(JSON_DIR, json_file), 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        height, width = data['imageHeight'], data['imageWidth']
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for shape in data['shapes']:
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
            
        # 形态学腐蚀操作
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        
        # 形态学相减，得到极细边缘
        edge_label = mask - eroded_mask
        
        # 保存 (乘以 255 是为了能在电脑上预览，但在训练时网络看到的是 0 和 255)
        save_path = os.path.join(SAVE_DIR, json_file.replace('.json', '_edge.png'))
        cv2.imwrite(save_path, edge_label * 255) 

    print(f"✅ 处理完成！极其锐利的物理边界图已全部保存至: {SAVE_DIR}")

if __name__ == '__main__':
    generate_edge_labels()
