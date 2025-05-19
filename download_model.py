#!/usr/bin/env python3
"""
下载声纹识别模型脚本
在构建Docker镜像前执行此脚本，将模型下载到本地
"""

import os
import sys
from modelscope.hub.snapshot_download import snapshot_download

# 设置SSL证书环境变量（解决SSL证书问题）
os.environ["SSL_CERT_FILE"] = ""
os.environ["HTTPX_SSL_VERIFY"] = "0"

# 模型ID
MODEL_ID = 'iic/speech_eres2netv2_sv_zh-cn_16k-common'
# 模型版本
MODEL_REVISION = 'v1.0.1'

def download_model():
    """下载模型到本地"""
    print(f"正在下载模型: {MODEL_ID}...")
    try:
        # 下载模型到当前目录下的modelscope_hub_cache文件夹
        model_dir = snapshot_download(
            model_id=MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir='./modelscope_hub_cache'
        )
        print(f"模型下载成功，保存在: {model_dir}")
        return True
    except Exception as e:
        print(f"模型下载失败: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    if not success:
        sys.exit(1)
    
    # 检查模型文件是否存在
    model_path = os.path.join(
        './modelscope_hub_cache', 
        MODEL_ID, 
        'pretrained_eres2netv2.ckpt'
    )
    
    if os.path.exists(model_path):
        print(f"模型文件确认存在: {model_path}")
        sys.exit(0)
    else:
        print(f"错误: 模型文件不存在: {model_path}")
        sys.exit(1) 