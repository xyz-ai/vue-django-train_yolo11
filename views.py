import os
import threading
import io
import json
import logging
import glob
from datetime import datetime
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
from .models import UploadedDataset
import torch
from contextlib import redirect_stdout
import zipfile

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量
CAPTURED_LOGS = []
TRAINING_STATUS = {
    "state": "idle",  # idle, running, completed, error
    "current_epoch": 0,
    "total_epochs": 50,
    "progress": 0.0
}

# 清除 CUDA_VISIBLE_DEVICES
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    del os.environ['CUDA_VISIBLE_DEVICES']

def find_data_yaml(dataset_root):
    """在 dataset_root 目录下递归查找 data.yaml 文件的完整路径"""
    for root, dirs, files in os.walk(dataset_root):
        if "data.yaml" in files:
            return os.path.join(root, "data.yaml")
    return None

@csrf_exempt
def upload_zip(request):
    """上传 ZIP 数据集并解压到 datasets/ 目录，同时记录上传记录到数据库"""
    if request.method == "POST":
        zip_file = request.FILES.get('zip_file')
        if not zip_file:
            return JsonResponse({"error": "没有上传 ZIP 文件"}, status=400)
        
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        
        folder_name = "upload_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = os.path.join(base_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            zip_bytes = io.BytesIO(zip_file.read())
            with zipfile.ZipFile(zip_bytes, 'r') as z:
                z.extractall(path=target_dir)
            UploadedDataset.objects.create(folder_name=folder_name)
            return JsonResponse({"message": "ZIP 数据集上传成功", "folder": folder_name})
        except Exception as e:
            logger.error(f"上传数据集失败: {str(e)}")
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "请求方式错误"}, status=400)

@csrf_exempt
def train_model(request):
    """启动训练任务，并支持用户选择训练轮数和模型"""
    global CAPTURED_LOGS, TRAINING_STATUS

    if request.method == "POST":
        try:
            data = json.loads(request.body)
            TRAINING_STATUS["total_epochs"] = int(data.get("epochs", 50))
            selected_model = data.get("model","yolo11s.pt")
            logger.info(f"接收到的模型选择: {selected_model}")
            TRAINING_STATUS["state"] = "running"
            TRAINING_STATUS["current_epoch"] = 0
            TRAINING_STATUS["progress"] = 0.0
            logger.info(f"训练开始，目标轮数: {TRAINING_STATUS['total_epochs']}, 模型: {selected_model}")

            latest_dataset = UploadedDataset.objects.order_by('-uploaded_at').first()
            if not latest_dataset:
                TRAINING_STATUS["state"] = "error"
                logger.error("没有可用的数据集")
                return JsonResponse({"error": "没有可用的数据集"}, status=400)

            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')
            latest_upload_dir = os.path.join(base_dir, latest_dataset.folder_name)
            data_yaml = find_data_yaml(latest_upload_dir)
            if not data_yaml:
                TRAINING_STATUS["state"] = "error"
                logger.error(f"在目录 {latest_upload_dir} 下找不到 data.yaml 文件")
                return JsonResponse({"error": f"在目录 {latest_upload_dir} 下找不到 data.yaml 文件"}, status=400)

            def background_training(data_yaml_path, model_name):
                logger.info(f"使用模型: {model_name}") 
                global CAPTURED_LOGS, TRAINING_STATUS
                CAPTURED_LOGS = []
                log_stream = io.StringIO()

                def on_train_epoch_end(trainer):
                    """回调函数，每轮训练结束时更新进度"""
                    epoch = trainer.epoch + 1
                    TRAINING_STATUS["current_epoch"] = epoch
                    TRAINING_STATUS["progress"] = (epoch / TRAINING_STATUS['total_epochs']) * 100
                    logger.info(f"回调触发 - 当前轮数: {epoch}, 进度: {TRAINING_STATUS['progress']}%")
                    CAPTURED_LOGS.extend(log_stream.getvalue().splitlines())
                    log_stream.truncate(0)
                    log_stream.seek(0)

                with redirect_stdout(log_stream):
                    try:
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        CAPTURED_LOGS.append(f"训练开始... 目标轮数: {TRAINING_STATUS['total_epochs']}")
                        CAPTURED_LOGS.append(f"使用设备: {device}, 模型: {model_name}")
                        model = YOLO(model_name)
                        model.add_callback("on_train_epoch_end", on_train_epoch_end)
                        results = model.train(
                            data=data_yaml_path,
                            epochs=TRAINING_STATUS["total_epochs"],
                            batch=4,
                            device=device,
                            verbose=True,
                            project='runs/train',
                            name=f'exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                            plots=True,  # 确保生成训练结果图片
                        )
                        # 适配新版 Ultralytics API
                        try:
                            mAP50 = results.box.map50 if hasattr(results, 'box') else 0
                        except AttributeError:
                            mAP50 = 0
                            logger.warning("无法获取 mAP50，可能由于分割任务或 API 变更")
                        log_message = f"训练完成！mAP50: {mAP50:.4f}"
                        CAPTURED_LOGS.append(log_message)
                        TRAINING_STATUS["state"] = "completed"
                        TRAINING_STATUS["progress"] = 100.0
                        logger.info("训练完成")
                    except Exception as e:
                        CAPTURED_LOGS.append(f"训练过程中出错: {str(e)}")
                        TRAINING_STATUS["state"] = "error"
                        logger.error(f"训练失败: {str(e)}")
                        raise
                    finally:
                        CAPTURED_LOGS.extend(log_stream.getvalue().splitlines())
                        log_stream.close()
                        logger.info(f"训练线程结束，日志条数: {len(CAPTURED_LOGS)}")

            training_thread = threading.Thread(target=background_training, args=(data_yaml, selected_model))
            training_thread.daemon = True
            training_thread.start()
            logger.info("训练线程已启动")

            return JsonResponse({"message": "训练开始了"})
        except Exception as e:
            TRAINING_STATUS["state"] = "error"
            logger.error(f"训练启动失败: {str(e)}")
            return JsonResponse({"error": f"训练启动失败: {str(e)}"}, status=500)

    return JsonResponse({"error": "请求方式错误"}, status=400)

@csrf_exempt
def training_logs(request):
    """返回当前训练日志"""
    global CAPTURED_LOGS
    logger.info(f"返回日志，条数: {len(CAPTURED_LOGS)}")
    return JsonResponse({"logs": CAPTURED_LOGS})

@csrf_exempt
def training_status(request):
    """返回训练状态和进度"""
    global TRAINING_STATUS
    logger.info(f"返回状态: {TRAINING_STATUS}")
    return JsonResponse(TRAINING_STATUS)

@csrf_exempt
def get_train_results(request):
    """返回最新训练结果目录中的图片 URL 列表"""
    if request.method == "GET":
        try:
            base_dir = os.path.join(os.getcwd(), "runs", "train")
            exp_dirs = [d for d in os.listdir(base_dir) if d.startswith("exp_")]
            if not exp_dirs:
                return JsonResponse({"error": "没有训练结果"}, status=400)
            
            latest_exp = max(exp_dirs, key=lambda d: os.path.getctime(os.path.join(base_dir, d)))
            exp_dir = os.path.join(base_dir, latest_exp)

            # 获取符合条件的图片文件
            image_paths = glob.glob(os.path.join(exp_dir, "*.jpg")) + glob.glob(os.path.join(exp_dir, "*.png"))
            if not image_paths:
                return JsonResponse({"error": "训练结果目录中没有图片"}, status=400)

            # **转换为前端可访问 URL**
            image_urls = [f"/runs/train/{latest_exp}/{os.path.basename(path)}" for path in image_paths]
            return JsonResponse({"images": image_urls[:10]})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "请求方式错误"}, status=400)
