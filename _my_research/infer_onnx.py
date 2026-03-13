import cv2
import numpy as np
import onnxruntime as ort
import os

# --- 配置区 ---
ONNX_PATH = "_my_research/output/onnx/rtdetrv2_r18vd_120e_coco.onnx"  # 确保路径正确
IMAGE_PATH = "_my_research/test.jpg"           # 替换为你的测试图片路径
OUTPUT_PATH = "_my_research/result.jpg"
CONF_THRES = 0.4                             # 置信度阈值
# 类别名称（根据你的数据集修改，保持顺序一致）
CLASSES = ['1', '2', '3', '4', '5', '6'] 

def preprocess(img_path, input_size=(640, 640)):
    """图像预处理：Resize + BGR2RGB + 归一化 + HWC2CHW"""
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        raise FileNotFoundError(f"无法读取图片: {img_path}")
        
    h, w = orig_img.shape[:2]
    # Resize 到模型输入尺寸
    img = cv2.resize(orig_img, input_size)
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 归一化 [0, 1]
    img = img.astype(np.float32) / 255.0
    # HWC -> CHW (BatchSize, Channel, Height, Width)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img, orig_img

def postprocess(outputs, orig_shape, conf_thres):
    """
    后处理：
    outputs[0] 是 logits 经过 sigmoid 后的结果，形状 [1, 300, 6]
    outputs[1] 是 boxes，形状 [1, 300, 4]，格式为 [cx, cy, w, h] (归一化值)
    """
    probs = outputs[0][0]  # [300, 6]
    boxes = outputs[1][0]  # [300, 4]
    
    orig_h, orig_w = orig_shape
    
    results = []
    for i in range(len(probs)):
        score = np.max(probs[i])
        if score > conf_thres:
            cls_id = np.argmax(probs[i])
            # 解构归一化坐标 [cx, cy, w, h]
            cx, cy, w, h = boxes[i]
            
            # 转换为左上角坐标 [x1, y1, x2, y2] 并还原到原图尺寸
            x1 = (cx - w / 2) * orig_w
            y1 = (cy - h / 2) * orig_h
            x2 = (cx + w / 2) * orig_w
            y2 = (cy + h / 2) * orig_h
            
            results.append([int(x1), int(y1), int(x2), int(y2), score, cls_id])
            
    return results

def main():
    # 1. 初始化推理引擎
    # 如果有 NVIDIA 显卡可使用 ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # 2. 预处理
    input_tensor, orig_img = preprocess(IMAGE_PATH)
    
    # 3. 执行推理
    # 你的模型现在只需要一个输入：images
    outputs = session.run(None, {input_name: input_tensor})
    
    # 4. 后处理 (注意：RT-DETR 通常不依赖传统的 NMS，模型输出已经包含 Top 300)
    results = postprocess(outputs, orig_img.shape[:2], CONF_THRES)
    
    # 5. 可视化绘制
    for res in results:
        x1, y1, x2, y2, score, cls_id = res
        label = f"{CLASSES[cls_id]}: {score:.2f}"
        print(f"检测到: {label} at [{x1}, {y1}, {x2}, {y2}]")
        
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig_img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite(OUTPUT_PATH, orig_img)
    print(f"结果已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()