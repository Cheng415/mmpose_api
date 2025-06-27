import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from mmpose.apis import MMPoseInferencer
from fastapi.responses import JSONResponse
from mmpose.utils import register_all_modules
import base64
import tempfile
import os

# 註冊所有模塊（必要的步驟）
register_all_modules()

# 創建 FastAPI 應用
app = FastAPI()

# 加入 CORS 中介軟體
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 設定哪些來源可以存取，"*" 代表全部允許，實際部署建議改成特定網域
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有 HTTP 方法，如 GET、POST、PUT
    allow_headers=["*"],  # 允許所有 headers
)

# 實例化 MMPoseInferencer
inferencer = MMPoseInferencer(
    pose2d='rtmo-s_8xb32-600e_coco-640x640',   # 2D姿勢模型config名或別名
    det_model='rtmo-s_8xb32-600e_coco-640x640',  # 物件偵測模型別名或config路徑
    device='cuda:0'
)

# numpy 不能直接傳回前端，因此需要清理
def clean_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: clean_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_numpy(i) for i in obj]
    else:
        return obj

# 影片預測路由
@app.post("/pose_video")
async def estimate_pose_video(file: UploadFile = File(...)):
    # 讀取影片文件
    video_bytes = await file.read()

    # 創建臨時檔案來儲存上傳的影片
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        video_path = temp_video_file.name
        temp_video_file.write(video_bytes)

    # 使用 cv2 打開臨時影片檔案
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Cannot open video file"}

    frames = []
    frame_idx = 0

    # 讀取影片中的每一幀
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += 1

    cap.release()

    # 批量預測
    result_gen = inferencer(frames, return_vis=False)
    frame_results = []

    for idx, result in enumerate(result_gen):
        predictions = result.get("predictions")
        cleaned_predictions = clean_numpy(predictions)

        frame_results.append({
            "frame_idx": idx,
            "predictions": cleaned_predictions,
            "visualization": None
        })

    # 刪除臨時檔案
    os.remove(video_path)

    # 返回結果
    return JSONResponse(content={"frames": frame_results})

