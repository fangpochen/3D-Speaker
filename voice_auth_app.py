import os
# 解决SSL证书问题
os.environ["SSL_CERT_FILE"] = ""
os.environ["HTTPX_SSL_VERIFY"] = "0"

import numpy as np
import torch
import torchaudio
import gradio as gr
import json
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download
from speakerlab.process.processor import FBank
from speakerlab.utils.builder import dynamic_import
import tempfile
import fastapi
from fastapi import UploadFile, File, Form
from starlette.responses import JSONResponse, FileResponse
import mysql.connector
from mysql.connector import pooling
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, HttpUrl
import requests
import io
import edge_tts
import asyncio
import uuid

# 响应模型定义
class ApiResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# 请求模型定义
class RegisterUserRequest(BaseModel):
    data: str  # URL to audio file
    user_id: str
    user_name: Optional[str] = None
    group_name: Optional[str] = "default"

class IdentifyUserRequest(BaseModel):
    data: str  # URL to audio file
    threshold: Optional[float] = 0.7  # 保留兼容性，实际不使用
    group_name: Optional[str] = "default"

class ListUsersRequest(BaseModel):
    group_name: Optional[str] = "default"

class DeleteUserRequest(BaseModel):
    user_id: str
    group_name: Optional[str] = "default"

# 组管理请求模型
class CreateGroupRequest(BaseModel):
    group_name: str
    description: Optional[str] = None

class DeleteGroupRequest(BaseModel):
    group_name: str
    force: Optional[bool] = False  # 是否强制删除（即使组内有用户）

class ListGroupsRequest(BaseModel):
    pass  # 无需参数

class GroupInfoRequest(BaseModel):
    group_name: str

# TTS请求模型
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "zh-CN-XiaoyiNeural"  # 默认使用中文女声
    rate: Optional[str] = "+0%"  # 语速调节
    volume: Optional[str] = "+0%"  # 音量调节
    pitch: Optional[str] = "+0Hz"  # 音调调节

class TTSVoicesRequest(BaseModel):
    language: Optional[str] = None  # 可选语言过滤

# 配置信息
MODEL_ID = 'iic/speech_eres2netv2_sv_zh-cn_16k-common'
model_config = {
    'obj': 'speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
        'baseWidth': 26,
        'scale': 2,
        'expansion': 2,
    },
}

# 数据库配置
DB_CONFIG = {
    'host': '106.13.0.50',
    'port': 3306,
    'user': 'root',
    'password': 'AaBb123456!',
    'database': 'voice_recognition'
}

# 创建数据库连接池
connection_pool = pooling.MySQLConnectionPool(
    pool_name="voice_auth_pool",
    pool_size=5,
    **DB_CONFIG
)

# 数据库路径（保留用于存储声纹文件）
DB_DIR = Path("/app/voice_auth_db") if os.path.exists("/app") else Path("voice_auth_db")
EMBEDDINGS_DIR = DB_DIR / "embeddings"
TTS_AUDIO_DIR = DB_DIR / "tts_audio"  # TTS生成的音频文件存储目录

# 创建目录
DB_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
TTS_AUDIO_DIR.mkdir(exist_ok=True)

# 初始化数据库表
def init_database():
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor()
        
        # 创建用户表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS voice_auth_users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            group_name VARCHAR(255) NOT NULL,
            user_id VARCHAR(255) NOT NULL,
            user_name VARCHAR(255),
            sample_id VARCHAR(255) NOT NULL,
            embedding_path VARCHAR(512) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uniq_group_user_sample (group_name, user_id, sample_id),
            INDEX idx_group_user (group_name, user_id)
        )
        ''')
        
        # 创建组信息表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS voice_auth_groups (
            id INT AUTO_INCREMENT PRIMARY KEY,
            group_name VARCHAR(255) NOT NULL UNIQUE,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_group_name (group_name)
        )
        ''')
        
        # 插入默认组（如果不存在）
        cursor.execute('''
        INSERT IGNORE INTO voice_auth_groups (group_name, description) 
        VALUES ('default', '默认分组')
        ''')
        
        conn.commit()
        print("数据库表初始化成功")
    except Exception as e:
        print(f"数据库表初始化失败: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 初始化数据库
init_database()

# 加载模型
def load_model():
    model_checkpoint_filename = 'pretrained_eres2netv2.ckpt'
    
    # 根据运行环境选择模型路径
    if os.path.exists("/app"):
        # Docker环境
        docker_model_root = "/app/modelscope_hub_cache"
        cache_dir_in_image = os.path.join(docker_model_root, MODEL_ID)
        model_path = os.path.join(cache_dir_in_image, model_checkpoint_filename)
        print(f"Docker环境: 尝试从路径加载模型: {model_path}")
    else:
        # 本地开发环境
        local_model_root = "./modelscope_hub_cache"
        cache_dir_local = os.path.join(local_model_root, MODEL_ID)
        model_path = os.path.join(cache_dir_local, model_checkpoint_filename)
        print(f"本地环境: 尝试从路径加载模型: {model_path}")

    if not os.path.exists(model_path):
        error_msg = (
            f"错误: 模型文件未找到: {model_path}\n"
            f"请确认以下事项:\n"
            f"1. 已运行 download_model.py 脚本下载模型到 'modelscope_hub_cache' 文件夹中。\n"
            f"2. 模型ID '{MODEL_ID}' 是正确的。\n"
            f"3. 检查点文件名 '{model_checkpoint_filename}' 是正确的。"
        )
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    embedding_model = dynamic_import(model_config['obj'])(**model_config['args'])
    embedding_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    embedding_model.to(device)
    embedding_model.eval()
    
    # 创建特征提取器
    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
    
    return embedding_model, feature_extractor, device

print("正在加载模型...")
model, feature_extractor, device = load_model()
print("模型加载完成!")

# 提取声纹特征
def extract_embedding(audio_path):
    # 加载音频
    try:
        wav, fs = torchaudio.load(audio_path)
        if fs != 16000:
            # 使用Resample替代sox_effects
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
            wav = resampler(wav)
            fs = 16000
            print(f"已将音频从 {fs}Hz 重采样到 16000Hz")
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
            
        # 提取特征
        feat = feature_extractor(wav).unsqueeze(0).to(device)
        
        # 提取声纹
        with torch.no_grad():
            embedding = model(feat).detach().squeeze(0).cpu().numpy()
            
        return embedding
    except Exception as e:
        print(f"提取声纹时发生错误: {e}")
        return None

# 计算相似度
def compute_similarity(emb1, emb2):
    sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    score = sim(torch.from_numpy(emb1).unsqueeze(0), 
                torch.from_numpy(emb2).unsqueeze(0)).item()
    return score

# 注册用户
def register_user(audio, user_id, user_name, group_name="default"):
    if not audio or not user_id:
        return "请提供音频文件和用户ID"
    
    user_id = user_id.strip()
    user_name = user_name.strip() if user_name else user_id
    group_name = group_name.strip() if group_name else "default"
    
    # 提取声纹
    embedding = extract_embedding(audio)
    if embedding is None:
        return "音频处理失败，请检查格式"
    
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor()
        
        # 检查分组是否存在，如果不存在则自动创建
        cursor.execute("SELECT COUNT(*) FROM voice_auth_groups WHERE group_name = %s", (group_name,))
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO voice_auth_groups (group_name, description) VALUES (%s, %s)",
                (group_name, f"自动创建的分组 {group_name}")
            )
            print(f"自动创建分组: {group_name}")
        
        # 查询用户在该组下的样本数
        cursor.execute(
            "SELECT COUNT(*) FROM voice_auth_users WHERE group_name = %s AND user_id = %s",
            (group_name, user_id)
        )
        sample_count = cursor.fetchone()[0]
        
        # 生成唯一的样本ID
        sample_id = f"{group_name}_{user_id}_sample_{sample_count + 1}"
        
        # 保存声纹（使用绝对路径）
        np_path = EMBEDDINGS_DIR / f"{sample_id}.npy"
        np.save(np_path, embedding)
        
        # 记录绝对路径到数据库
        abs_path = str(np_path.absolute())
        print(f"保存声纹文件到: {abs_path}")
        
        # 插入数据库
        cursor.execute(
            """
            INSERT INTO voice_auth_users 
            (group_name, user_id, user_name, sample_id, embedding_path) 
            VALUES (%s, %s, %s, %s, %s)
            """,
            (group_name, user_id, user_name, sample_id, abs_path)
        )
        
        conn.commit()
        return f"成功注册用户 {user_name}(ID: {user_id}, 组: {group_name})，声纹样本 #{sample_count + 1}"
    
    except Exception as e:
        print(f"注册用户时发生错误: {e}")
        return f"注册失败: {str(e)}"
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 识别用户
def identify_user(audio, threshold=0.70, group_name="default"):
    if not audio:
        return {
            "success": False,
            "message": "请提供音频文件",
            "data": None,
            "error": "缺少音频文件"
        }
    
    group_name = group_name.strip() if group_name else "default"
    
    # 提取声纹
    current_embedding = extract_embedding(audio)
    if current_embedding is None:
        return {
            "success": False,
            "message": "音频处理失败，请检查格式",
            "data": None,
            "error": "音频处理失败"
        }
    
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 检查该组下是否有用户
        cursor.execute(
            "SELECT COUNT(*) as count FROM voice_auth_users WHERE group_name = %s",
            (group_name,)
        )
        result = cursor.fetchone()
        if result['count'] == 0:
            return {
                "success": False,
                "message": f"分组 '{group_name}' 下没有注册用户，请先注册",
                "data": None,
                "error": "分组无用户"
            }
        
        # 获取该组下所有用户的声纹样本
        cursor.execute(
            """
            SELECT user_id, user_name, sample_id, embedding_path 
            FROM voice_auth_users 
            WHERE group_name = %s
            """,
            (group_name,)
        )
        
        max_score = -1.0  # 初始化为-1，确保能找到最大值
        matched_user_info = None
        
        # 计算与所有样本的相似度
        for record in cursor.fetchall():
            embedding_path = record['embedding_path']
            
            # 处理路径：如果是相对路径，转换为绝对路径
            if not os.path.isabs(embedding_path):
                embedding_path = os.path.join(os.getcwd(), embedding_path)
            
            print(f"尝试加载声纹文件: {embedding_path}")
            
            # 检查文件是否存在
            if os.path.exists(embedding_path):
                try:
                    sample_embedding = np.load(embedding_path)
                    score = compute_similarity(current_embedding, sample_embedding)
                    print(f"用户: {record['user_id']}, 样本: {record['sample_id']}, 相似度: {score:.4f}")
                    if score > max_score:
                        max_score = score
                        matched_user_info = record
                except Exception as e:
                    print(f"加载文件 {embedding_path} 或计算相似度出错: {e}")
            else:
                print(f"声纹文件不存在: {embedding_path}")
        
        # 返回结果 - 直接返回最相似的用户，不使用阈值判断
        if matched_user_info:
            user_data = {
                "user_id": matched_user_info['user_id'],
                "user_name": matched_user_info['user_name'],
                "group_name": group_name,
                "score": float(f"{max_score:.4f}"),
                "threshold": float(threshold)  # 保留threshold字段以保持API兼容性
            }
            return {
                "success": True,
                "message": f"✅ 识别完成! 最相似用户: {matched_user_info['user_name']} (ID: {matched_user_info['user_id']}, 组: {group_name})\n相似度分数: {max_score:.4f}",
                "data": user_data,
                "error": None
            }
        else:
            return {
                "success": False,
                "message": f"❌ 无法识别用户，可能所有声纹文件都无法加载",
                "data": {
                    "threshold": float(threshold),
                    "max_score": float(f"{max_score:.4f}")
                },
                "error": "无有效声纹文件"
            }
    
    except Exception as e:
        print(f"识别用户时发生错误: {e}")
        return {
            "success": False,
            "message": "识别过程中发生内部错误",
            "data": None,
            "error": str(e)
        }
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 列出所有用户
def list_users(group_name="default"):
    group_name = group_name.strip() if group_name else "default"
    
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 查询指定分组的用户
        cursor.execute(
            """
            SELECT user_id, user_name, COUNT(sample_id) as sample_count 
            FROM voice_auth_users 
            WHERE group_name = %s 
            GROUP BY user_id, user_name
            """,
            (group_name,)
        )
        
        users = cursor.fetchall()
        
        if not users:
            return f"分组 '{group_name}' 下没有注册用户"
        
        result = f"分组 '{group_name}' 已注册用户列表:\n"
        for user in users:
            result += f"- {user['user_name']} (ID: {user['user_id']}), 样本数: {user['sample_count']}\n"
        
        return result
    
    except Exception as e:
        print(f"列出用户时发生错误: {e}")
        return f"获取用户列表失败: {str(e)}"
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 删除用户
def delete_user(user_id, group_name="default"):
    if not user_id:
        return "请提供用户ID"
    
    group_name = group_name.strip() if group_name else "default"
    
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 检查用户是否存在
        cursor.execute(
            """
            SELECT user_name, sample_id, embedding_path 
            FROM voice_auth_users 
            WHERE group_name = %s AND user_id = %s
            """,
            (group_name, user_id)
        )
        
        samples = cursor.fetchall()
        
        if not samples:
            return f"分组 '{group_name}' 下不存在ID为 '{user_id}' 的用户"
        
        user_name = samples[0]['user_name']
        
        # 删除声纹文件
        for sample in samples:
            embedding_path = sample['embedding_path']
            if os.path.exists(embedding_path):
                print(f"删除声纹文件: {embedding_path}")
                os.remove(embedding_path)
        
        # 从数据库中删除
        cursor.execute(
            "DELETE FROM voice_auth_users WHERE group_name = %s AND user_id = %s",
            (group_name, user_id)
        )
        
        conn.commit()
        
        return f"已删除用户 {user_name} (ID: {user_id}, 组: {group_name}) 及其所有声纹样本"
    
    except Exception as e:
        print(f"删除用户时发生错误: {e}")
        return f"删除失败: {str(e)}"
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 组管理函数
def create_group(group_name, description=None):
    """创建新的分组"""
    if not group_name:
        return {"success": False, "message": "请提供分组名称", "error": "缺少分组名称"}
    
    group_name = group_name.strip()
    if not group_name:
        return {"success": False, "message": "分组名称不能为空", "error": "分组名称无效"}
    
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor()
        
        # 检查分组是否已存在
        cursor.execute("SELECT COUNT(*) FROM voice_auth_groups WHERE group_name = %s", (group_name,))
        if cursor.fetchone()[0] > 0:
            return {"success": False, "message": f"分组 '{group_name}' 已存在", "error": "分组已存在"}
        
        # 创建新分组
        cursor.execute(
            "INSERT INTO voice_auth_groups (group_name, description) VALUES (%s, %s)",
            (group_name, description or f"分组 {group_name}")
        )
        
        conn.commit()
        return {"success": True, "message": f"成功创建分组 '{group_name}'"}
    
    except Exception as e:
        print(f"创建分组时发生错误: {e}")
        return {"success": False, "message": "创建分组失败", "error": str(e)}
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def delete_group(group_name, force=False):
    """删除分组"""
    if not group_name:
        return {"success": False, "message": "请提供分组名称", "error": "缺少分组名称"}
    
    group_name = group_name.strip()
    if group_name == "default":
        return {"success": False, "message": "不能删除默认分组", "error": "默认分组受保护"}
    
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 检查分组是否存在
        cursor.execute("SELECT COUNT(*) as count FROM voice_auth_groups WHERE group_name = %s", (group_name,))
        result = cursor.fetchone()
        if result['count'] == 0:
            return {"success": False, "message": f"分组 '{group_name}' 不存在", "error": "分组不存在"}
        
        # 检查分组下是否有用户
        cursor.execute("SELECT COUNT(*) as count FROM voice_auth_users WHERE group_name = %s", (group_name,))
        user_count = cursor.fetchone()['count']
        
        if user_count > 0 and not force:
            return {
                "success": False, 
                "message": f"分组 '{group_name}' 下还有 {user_count} 个用户，请先删除用户或使用强制删除", 
                "error": "分组非空"
            }
        
        # 如果强制删除，先删除分组下的所有用户
        if force and user_count > 0:
            # 获取所有用户的声纹文件路径
            cursor.execute(
                "SELECT embedding_path FROM voice_auth_users WHERE group_name = %s",
                (group_name,)
            )
            embedding_paths = cursor.fetchall()
            
            # 删除声纹文件
            for record in embedding_paths:
                embedding_path = record['embedding_path']
                if os.path.exists(embedding_path):
                    try:
                        os.remove(embedding_path)
                        print(f"删除声纹文件: {embedding_path}")
                    except Exception as e:
                        print(f"删除声纹文件失败 {embedding_path}: {e}")
            
            # 删除用户记录
            cursor.execute("DELETE FROM voice_auth_users WHERE group_name = %s", (group_name,))
        
        # 删除分组
        cursor.execute("DELETE FROM voice_auth_groups WHERE group_name = %s", (group_name,))
        
        conn.commit()
        
        if force and user_count > 0:
            return {"success": True, "message": f"已强制删除分组 '{group_name}' 及其下的 {user_count} 个用户"}
        else:
            return {"success": True, "message": f"已删除分组 '{group_name}'"}
    
    except Exception as e:
        print(f"删除分组时发生错误: {e}")
        return {"success": False, "message": "删除分组失败", "error": str(e)}
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def list_groups():
    """列出所有分组"""
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 获取所有分组及其用户数量
        cursor.execute("""
            SELECT 
                g.group_name, 
                g.description, 
                g.created_at,
                COUNT(u.user_id) as user_count
            FROM voice_auth_groups g
            LEFT JOIN (
                SELECT group_name, user_id 
                FROM voice_auth_users 
                GROUP BY group_name, user_id
            ) u ON g.group_name = u.group_name
            GROUP BY g.group_name, g.description, g.created_at
            ORDER BY g.created_at
        """)
        
        groups = cursor.fetchall()
        
        if not groups:
            return {"success": True, "message": "暂无分组", "data": []}
        
        # 格式化结果
        group_list = []
        for group in groups:
            group_list.append({
                "group_name": group['group_name'],
                "description": group['description'],
                "user_count": group['user_count'],
                "created_at": group['created_at'].strftime("%Y-%m-%d %H:%M:%S") if group['created_at'] else None
            })
        
        return {"success": True, "message": f"共找到 {len(groups)} 个分组", "data": group_list}
    
    except Exception as e:
        print(f"列出分组时发生错误: {e}")
        return {"success": False, "message": "获取分组列表失败", "error": str(e)}
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

def get_group_info(group_name):
    """获取分组详细信息"""
    if not group_name:
        return {"success": False, "message": "请提供分组名称", "error": "缺少分组名称"}
    
    group_name = group_name.strip()
    
    try:
        conn = connection_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        
        # 获取分组基本信息
        cursor.execute(
            "SELECT group_name, description, created_at FROM voice_auth_groups WHERE group_name = %s",
            (group_name,)
        )
        group_info = cursor.fetchone()
        
        if not group_info:
            return {"success": False, "message": f"分组 '{group_name}' 不存在", "error": "分组不存在"}
        
        # 获取用户统计信息
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT user_id) as user_count,
                COUNT(*) as sample_count
            FROM voice_auth_users 
            WHERE group_name = %s
        """, (group_name,))
        
        stats = cursor.fetchone()
        
        # 获取用户列表
        cursor.execute("""
            SELECT user_id, user_name, COUNT(*) as sample_count
            FROM voice_auth_users 
            WHERE group_name = %s
            GROUP BY user_id, user_name
            ORDER BY user_name
        """, (group_name,))
        
        users = cursor.fetchall()
        
        result = {
            "group_name": group_info['group_name'],
            "description": group_info['description'],
            "created_at": group_info['created_at'].strftime("%Y-%m-%d %H:%M:%S") if group_info['created_at'] else None,
            "user_count": stats['user_count'] or 0,
            "total_samples": stats['sample_count'] or 0,
            "users": [
                {
                    "user_id": user['user_id'],
                    "user_name": user['user_name'],
                    "sample_count": user['sample_count']
                }
                for user in users
            ]
        }
        
        return {"success": True, "message": f"分组 '{group_name}' 信息", "data": result}
    
    except Exception as e:
        print(f"获取分组信息时发生错误: {e}")
        return {"success": False, "message": "获取分组信息失败", "error": str(e)}
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# 下载URL音频文件
async def download_audio_from_url(url: str):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        temp_audio_path = Path(temp_dir) / "audio_file.wav"
        
        # 保存音频文件
        with open(temp_audio_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return str(temp_audio_path), temp_dir
    except Exception as e:
        print(f"下载音频文件时发生错误: {e}")
        raise Exception(f"下载音频文件失败: {str(e)}")

# TTS核心功能函数
async def text_to_speech(text: str, voice: str = "zh-CN-XiaoyiNeural", 
                        rate: str = "+0%", volume: str = "+0%", pitch: str = "+0Hz"):
    """
    使用edge-tts将文本转换为语音
    """
    try:
        # 生成唯一的文件名
        audio_filename = f"tts_{uuid.uuid4().hex}.mp3"
        audio_path = TTS_AUDIO_DIR / audio_filename
        
        # 创建edge-tts的Communicate对象
        communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume, pitch=pitch)
        
        # 保存音频文件
        await communicate.save(str(audio_path))
        
        return str(audio_path), audio_filename
    except Exception as e:
        print(f"TTS转换时发生错误: {e}")
        raise Exception(f"TTS转换失败: {str(e)}")

async def get_available_voices(language_filter: str = None):
    """
    获取可用的TTS语音列表
    """
    try:
        voices = await edge_tts.list_voices()
        
        if language_filter:
            # 过滤指定语言的语音
            filtered_voices = [
                voice for voice in voices 
                if language_filter.lower() in voice["Locale"].lower()
            ]
            return filtered_voices
        
        return voices
    except Exception as e:
        print(f"获取语音列表时发生错误: {e}")
        return []

def get_voice_options():
    """
    获取常用语音选项（同步版本，用于Gradio）
    """
    common_voices = {
        "中文女声(晓伊)": "zh-CN-XiaoyiNeural",
        "中文男声(云扬)": "zh-CN-YunyangNeural", 
        "中文女声(晓晓)": "zh-CN-XiaoxiaoNeural",
        "中文男声(云希)": "zh-CN-YunxiNeural",
        "英文女声(Aria)": "en-US-AriaNeural",
        "英文男声(Davis)": "en-US-DavisNeural",
        "日语女声(Nanami)": "ja-JP-NanamiNeural",
        "韩语女声(Sun-Hi)": "ko-KR-SunHiNeural"
    }
    return common_voices

def tts_gradio_wrapper(text: str, voice_name: str, rate: float, volume: float, pitch: float):
    """
    Gradio包装器函数（同步版本）
    """
    if not text.strip():
        return None, "请输入要转换的文本"
    
    try:
        # 获取语音选项
        voice_options = get_voice_options()
        voice_code = voice_options.get(voice_name, "zh-CN-XiaoyiNeural")
        
        # 格式化参数
        rate_str = f"+{int(rate)}%" if rate >= 0 else f"{int(rate)}%"
        volume_str = f"+{int(volume)}%" if volume >= 0 else f"{int(volume)}%"
        pitch_str = f"+{int(pitch)}Hz" if pitch >= 0 else f"{int(pitch)}Hz"
        
        # 在新的事件循环中运行异步函数
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            audio_path, filename = loop.run_until_complete(
                text_to_speech(text, voice_code, rate_str, volume_str, pitch_str)
            )
            return audio_path, f"✅ TTS转换成功！音频文件: {filename}"
        finally:
            loop.close()
    except Exception as e:
        return None, f"❌ TTS转换失败: {str(e)}"

# 自定义API路由处理函数
def add_custom_routes(fastapi_app):
    @fastapi_app.post("/direct/register_user", 
                     summary="注册用户声纹",
                     description="上传音频文件，注册用户声纹信息，支持分组管理",
                     response_model=ApiResponse)
    async def direct_register_user(
        data: UploadFile = File(..., description="音频文件，支持各种常见音频格式"),
        user_id: str = Form(..., description="用户ID，必填"),
        user_name: str = Form(None, description="用户姓名，选填，默认与用户ID相同"),
        group_name: str = Form("default", description="分组名称，选填，默认为'default'")
    ):
        try:
            # 保存上传的文件到临时目录
            temp_dir = tempfile.mkdtemp()
            temp_audio_path = Path(temp_dir) / data.filename
            
            with open(temp_audio_path, "wb") as f:
                f.write(await data.read())
            
            # 调用注册函数
            result = register_user(str(temp_audio_path), user_id, user_name or user_id, group_name)
            
            # 清理临时文件
            os.remove(temp_audio_path)
            os.rmdir(temp_dir)
            
            # 返回结果
            return ApiResponse(success=True, message=result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    @fastapi_app.post("/direct/identify_user",
                     summary="识别用户声纹",
                     description="上传音频文件，返回最相似的已注册用户（不使用阈值判断）",
                     response_model=ApiResponse)
    async def direct_identify_user(
        data: UploadFile = File(..., description="音频文件，支持各种常见音频格式"),
        threshold: float = Form(0.7, description="阈值参数（保留兼容性，实际不使用）"),
        group_name: str = Form("default", description="分组名称，选填，默认为'default'")
    ):
        temp_audio_path = None
        temp_dir = None
        try:
            # 保存上传的文件到临时目录
            temp_dir = tempfile.mkdtemp()
            temp_audio_path = Path(temp_dir) / data.filename
            
            with open(temp_audio_path, "wb") as f:
                f.write(await data.read())
            
            # 调用识别函数
            result_dict = identify_user(str(temp_audio_path), threshold, group_name)
            
            # 返回结果
            return ApiResponse(**result_dict)
        except Exception as e:
            return ApiResponse(success=False, message="处理请求时发生内部错误", error=str(e), data=None)
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
    
    @fastapi_app.post("/direct/list_users",
                     summary="列出已注册用户",
                     description="列出指定分组下所有已注册的用户及其声纹样本数量",
                     response_model=ApiResponse)
    async def direct_list_users(group_name: str = Form("default", description="分组名称，选填，默认为'default'")):
        try:
            result = list_users(group_name)
            return ApiResponse(success=True, message=result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    @fastapi_app.post("/direct/delete_user",
                     summary="删除用户声纹",
                     description="删除指定分组下的用户及其所有声纹样本",
                     response_model=ApiResponse)
    async def direct_delete_user(
        user_id: str = Form(..., description="要删除的用户ID"),
        group_name: str = Form("default", description="分组名称，选填，默认为'default'")
    ):
        try:
            result = delete_user(user_id, group_name)
            return ApiResponse(success=True, message=result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
            
    # 新增JSON格式的API接口
    @fastapi_app.post("/api/register_user", 
                     summary="注册用户声纹(JSON格式)",
                     description="通过URL注册用户声纹信息，支持分组管理",
                     response_model=ApiResponse)
    async def api_register_user(request: RegisterUserRequest):
        try:
            # 下载音频文件
            temp_audio_path, temp_dir = await download_audio_from_url(request.data)
            
            # 调用注册函数
            result = register_user(
                temp_audio_path, 
                request.user_id, 
                request.user_name or request.user_id, 
                request.group_name
            )
            
            # 清理临时文件
            os.remove(temp_audio_path)
            os.rmdir(temp_dir)
            
            # 返回结果
            return ApiResponse(success=True, message=result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    @fastapi_app.post("/api/identify_user",
                     summary="识别用户声纹(JSON格式)",
                     description="通过URL识别声纹，返回最相似的已注册用户（不使用阈值判断）",
                     response_model=ApiResponse)
    async def api_identify_user(request: IdentifyUserRequest):
        temp_audio_path = None
        temp_dir = None
        try:
            # 下载音频文件
            temp_audio_path, temp_dir = await download_audio_from_url(request.data)
            
            # 调用识别函数
            result_dict = identify_user(temp_audio_path, request.threshold, request.group_name)
            
            # 返回结果
            return ApiResponse(**result_dict)
        except Exception as e:
            # 区分下载错误和识别错误
            if "下载音频文件失败" in str(e):
                 return ApiResponse(success=False, message=str(e), error="音频下载失败", data=None)
            return ApiResponse(success=False, message="处理请求时发生内部错误", error=str(e), data=None)
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
    
    @fastapi_app.post("/api/list_users",
                     summary="列出已注册用户(JSON格式)",
                     description="列出指定分组下所有已注册的用户及其声纹样本数量",
                     response_model=ApiResponse)
    async def api_list_users(request: ListUsersRequest):
        try:
            result = list_users(request.group_name)
            return ApiResponse(success=True, message=result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    @fastapi_app.post("/api/delete_user",
                     summary="删除用户声纹(JSON格式)",
                     description="删除指定分组下的用户及其所有声纹样本",
                     response_model=ApiResponse)
    async def api_delete_user(request: DeleteUserRequest):
        try:
            result = delete_user(request.user_id, request.group_name)
            return ApiResponse(success=True, message=result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    # 组管理API接口
    @fastapi_app.post("/api/create_group",
                     summary="创建分组",
                     description="创建新的用户分组",
                     response_model=ApiResponse)
    async def api_create_group(request: CreateGroupRequest):
        try:
            result = create_group(request.group_name, request.description)
            return ApiResponse(**result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    @fastapi_app.post("/api/delete_group",
                     summary="删除分组",
                     description="删除指定分组，可选择是否强制删除（包括分组下的所有用户）",
                     response_model=ApiResponse)
    async def api_delete_group(request: DeleteGroupRequest):
        try:
            result = delete_group(request.group_name, request.force)
            return ApiResponse(**result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    @fastapi_app.post("/api/list_groups",
                     summary="列出所有分组",
                     description="获取系统中所有分组的列表及其基本信息",
                     response_model=ApiResponse)
    async def api_list_groups(request: ListGroupsRequest):
        try:
            result = list_groups()
            return ApiResponse(**result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    @fastapi_app.post("/api/group_info",
                     summary="获取分组详细信息",
                     description="获取指定分组的详细信息，包括用户列表和统计数据",
                     response_model=ApiResponse)
    async def api_group_info(request: GroupInfoRequest):
        try:
            result = get_group_info(request.group_name)
            return ApiResponse(**result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    # 直接上传文件的组管理接口
    @fastapi_app.post("/direct/create_group",
                     summary="创建分组(表单格式)",
                     description="通过表单创建新的用户分组",
                     response_model=ApiResponse)
    async def direct_create_group(
        group_name: str = Form(..., description="分组名称"),
        description: str = Form(None, description="分组描述，选填")
    ):
        try:
            result = create_group(group_name, description)
            return ApiResponse(**result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    @fastapi_app.post("/direct/delete_group",
                     summary="删除分组(表单格式)",
                     description="通过表单删除指定分组",
                     response_model=ApiResponse)
    async def direct_delete_group(
        group_name: str = Form(..., description="要删除的分组名称"),
        force: bool = Form(False, description="是否强制删除（包括分组下的所有用户）")
    ):
        try:
            result = delete_group(group_name, force)
            return ApiResponse(**result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    @fastapi_app.get("/direct/list_groups",
                    summary="列出所有分组(GET请求)",
                    description="获取系统中所有分组的列表及其基本信息",
                    response_model=ApiResponse)
    async def direct_list_groups():
        try:
            result = list_groups()
            return ApiResponse(**result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    @fastapi_app.get("/direct/group_info/{group_name}",
                    summary="获取分组详细信息(GET请求)",
                    description="获取指定分组的详细信息",
                    response_model=ApiResponse)
    async def direct_group_info(group_name: str):
        try:
            result = get_group_info(group_name)
            return ApiResponse(**result)
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    # TTS API接口
    @fastapi_app.post("/api/tts",
                     summary="文本转语音",
                     description="将输入的文本转换为语音文件",
                     response_model=ApiResponse)
    async def api_text_to_speech(request: TTSRequest):
        try:
            audio_path, filename = await text_to_speech(
                request.text, 
                request.voice, 
                request.rate, 
                request.volume, 
                request.pitch
            )
            
            # 返回文件信息
            return ApiResponse(
                success=True, 
                message="TTS转换成功",
                data={
                    "filename": filename,
                    "audio_path": audio_path,
                    "download_url": f"/api/tts/download/{filename}"
                }
            )
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    @fastapi_app.get("/api/tts/download/{filename}",
                    summary="下载TTS音频文件",
                    description="下载指定的TTS生成的音频文件")
    async def download_tts_audio(filename: str):
        try:
            file_path = TTS_AUDIO_DIR / filename
            if not file_path.exists():
                return JSONResponse(
                    status_code=404, 
                    content={"error": "文件不存在"}
                )
            
            return FileResponse(
                path=str(file_path),
                filename=filename,
                media_type="audio/mpeg"
            )
        except Exception as e:
            return JSONResponse(
                status_code=500, 
                content={"error": str(e)}
            )
    
    @fastapi_app.post("/api/tts/voices",
                     summary="获取可用语音列表",
                     description="获取edge-tts支持的所有语音列表",
                     response_model=ApiResponse)
    async def api_get_voices(request: TTSVoicesRequest):
        try:
            voices = await get_available_voices(request.language)
            return ApiResponse(
                success=True,
                message=f"获取到 {len(voices)} 个语音",
                data={"voices": voices}
            )
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    @fastapi_app.get("/api/tts/voices",
                    summary="获取可用语音列表(GET)",
                    description="获取edge-tts支持的所有语音列表",
                    response_model=ApiResponse)
    async def api_get_voices_get(language: str = None):
        try:
            voices = await get_available_voices(language)
            return ApiResponse(
                success=True,
                message=f"获取到 {len(voices)} 个语音",
                data={"voices": voices}
            )
        except Exception as e:
            return ApiResponse(success=False, error=str(e))
    
    # 直接上传的TTS接口
    @fastapi_app.post("/direct/tts",
                     summary="文本转语音(表单格式)",
                     description="通过表单提交文本进行TTS转换",
                     response_model=ApiResponse)
    async def direct_text_to_speech(
        text: str = Form(..., description="要转换的文本"),
        voice: str = Form("zh-CN-XiaoyiNeural", description="语音类型"),
        rate: str = Form("+0%", description="语速调节，如：+20%、-10%"),
        volume: str = Form("+0%", description="音量调节，如：+20%、-10%"),
        pitch: str = Form("+0Hz", description="音调调节，如：+50Hz、-20Hz")
    ):
        try:
            audio_path, filename = await text_to_speech(text, voice, rate, volume, pitch)
            
            return ApiResponse(
                success=True,
                message="TTS转换成功",
                data={
                    "filename": filename,
                    "audio_path": audio_path,
                    "download_url": f"/api/tts/download/{filename}"
                }
            )
        except Exception as e:
            return ApiResponse(success=False, error=str(e))

# 创建Gradio界面
with gr.Blocks(title="声纹识别系统") as app:
    gr.Markdown("# aying啊应 声纹识别系统")
    
    with gr.Tab("声纹注册"):
        with gr.Row():
            with gr.Column():
                reg_audio = gr.Audio(type="filepath", label="录制或上传声音")
                reg_group_name = gr.Textbox(label="分组名称", placeholder="默认为default", value="default")
                reg_user_id = gr.Textbox(label="用户ID (必填)", placeholder="例如: 001")
                reg_user_name = gr.Textbox(label="用户姓名 (选填)", placeholder="例如: 张三")
                reg_btn = gr.Button("注册声纹", variant="primary")
            with gr.Column():
                reg_output = gr.Textbox(label="注册结果", lines=3)
                users_list = gr.Textbox(label="当前注册用户", lines=10)
                refresh_group = gr.Textbox(label="查询分组", placeholder="输入要查询的分组名称", value="default")
                refresh_btn = gr.Button("刷新用户列表")
    
    with gr.Tab("声纹识别"):
        with gr.Row():
            with gr.Column():
                ident_audio = gr.Audio(type="filepath", label="录制或上传声音")
                ident_group = gr.Textbox(label="分组名称", placeholder="默认为default", value="default")
                threshold = gr.Slider(0.5, 0.95, value=0.7, step=0.01, label="阈值（仅显示）", info="当前版本直接返回最相似用户，不使用阈值判断")
                ident_btn = gr.Button("识别声纹", variant="primary")
            with gr.Column():
                ident_output = gr.Textbox(label="识别结果", lines=5)
    
    with gr.Tab("用户管理"):
        with gr.Row():
            with gr.Column():
                del_group = gr.Textbox(label="分组名称", placeholder="默认为default", value="default")
                del_user_id = gr.Textbox(label="用户ID", placeholder="要删除的用户ID")
                del_btn = gr.Button("删除用户", variant="stop")
                del_output = gr.Textbox(label="操作结果", lines=3)
    
    with gr.Tab("分组管理"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 创建分组")
                create_group_name = gr.Textbox(label="分组名称", placeholder="输入新分组名称")
                create_group_desc = gr.Textbox(label="分组描述", placeholder="输入分组描述（可选）")
                create_group_btn = gr.Button("创建分组", variant="primary")
                create_group_output = gr.Textbox(label="创建结果", lines=2)
                
                gr.Markdown("### 删除分组")
                delete_group_name = gr.Textbox(label="分组名称", placeholder="要删除的分组名称")
                delete_group_force = gr.Checkbox(label="强制删除（包括分组下的所有用户）", value=False)
                delete_group_btn = gr.Button("删除分组", variant="stop")
                delete_group_output = gr.Textbox(label="删除结果", lines=2)
            
            with gr.Column():
                gr.Markdown("### 分组列表")
                groups_list = gr.Textbox(label="所有分组", lines=15)
                refresh_groups_btn = gr.Button("刷新分组列表")
                
                gr.Markdown("### 分组详情")
                group_info_name = gr.Textbox(label="分组名称", placeholder="输入要查询的分组名称")
                group_info_btn = gr.Button("查询分组详情")
                group_info_output = gr.Textbox(label="分组详情", lines=8)
    
    with gr.Tab("文本转语音(TTS)"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### TTS设置")
                tts_text = gr.Textbox(
                    label="输入文本", 
                    placeholder="请输入要转换为语音的文本...",
                    lines=5
                )
                
                voice_options = get_voice_options()
                tts_voice = gr.Dropdown(
                    choices=list(voice_options.keys()),
                    value="中文女声(晓伊)",
                    label="选择语音"
                )
                
                with gr.Row():
                    tts_rate = gr.Slider(
                        minimum=-50, maximum=100, value=0, step=5,
                        label="语速调节(%)", 
                        info="负数=慢，正数=快"
                    )
                    tts_volume = gr.Slider(
                        minimum=-50, maximum=100, value=0, step=5,
                        label="音量调节(%)",
                        info="负数=小声，正数=大声"
                    )
                    tts_pitch = gr.Slider(
                        minimum=-100, maximum=100, value=0, step=10,
                        label="音调调节(Hz)",
                        info="负数=低音，正数=高音"
                    )
                
                tts_btn = gr.Button("生成语音", variant="primary", size="lg")
                
            with gr.Column():
                gr.Markdown("### 生成结果")
                tts_audio = gr.Audio(label="生成的语音", interactive=False)
                tts_output = gr.Textbox(label="状态信息", lines=3)
                
                gr.Markdown("### 使用说明")
                gr.Markdown("""
                **TTS功能说明:**
                - 支持中文、英文、日语、韩语等多种语言
                - 可调节语速、音量和音调
                - 生成的音频文件会保存在服务器上
                - 支持API调用：`POST /api/tts`
                
                **API示例:**
                ```json
                {
                    "text": "你好，世界！",
                    "voice": "zh-CN-XiaoyiNeural",
                    "rate": "+20%",
                    "volume": "+10%",
                    "pitch": "+0Hz"
                }
                ```
                """)
    
    # 功能连接
    reg_btn.click(register_user, inputs=[reg_audio, reg_user_id, reg_user_name, reg_group_name], outputs=reg_output)
    refresh_btn.click(list_users, inputs=[refresh_group], outputs=users_list)
    
    # 应用加载时执行
    def on_app_load():
        return list_users("default")
    
    app.load(on_app_load, inputs=None, outputs=users_list)

    # 修改Gradio界面identify_btn的click事件，以处理新的返回格式
    def gradio_identify_wrapper(audio, threshold, group_name):
        result = identify_user(audio, threshold, group_name)
        return result.get("message", "处理出错")

    ident_btn.click(gradio_identify_wrapper, inputs=[ident_audio, threshold, ident_group], outputs=ident_output)
    del_btn.click(delete_user, inputs=[del_user_id, del_group], outputs=del_output)
    
    # 组管理功能连接
    def gradio_create_group_wrapper(group_name, description):
        result = create_group(group_name, description)
        return result.get("message", "处理出错")
    
    def gradio_delete_group_wrapper(group_name, force):
        result = delete_group(group_name, force)
        return result.get("message", "处理出错")
    
    def gradio_list_groups_wrapper():
        result = list_groups()
        if result["success"]:
            if result["data"]:
                groups_text = f"{result['message']}\n\n"
                for group in result["data"]:
                    groups_text += f"📁 {group['group_name']}\n"
                    groups_text += f"   描述: {group['description']}\n"
                    groups_text += f"   用户数: {group['user_count']}\n"
                    groups_text += f"   创建时间: {group['created_at']}\n\n"
                return groups_text
            else:
                return result["message"]
        else:
            return f"错误: {result.get('error', '未知错误')}"
    
    def gradio_group_info_wrapper(group_name):
        result = get_group_info(group_name)
        if result["success"]:
            data = result["data"]
            info_text = f"分组名称: {data['group_name']}\n"
            info_text += f"描述: {data['description']}\n"
            info_text += f"创建时间: {data['created_at']}\n"
            info_text += f"用户数量: {data['user_count']}\n"
            info_text += f"声纹样本总数: {data['total_samples']}\n\n"
            
            if data['users']:
                info_text += "用户列表:\n"
                for user in data['users']:
                    info_text += f"- {user['user_name']} (ID: {user['user_id']}) - {user['sample_count']} 个样本\n"
            else:
                info_text += "该分组下暂无用户"
            
            return info_text
        else:
            return f"错误: {result.get('error', '未知错误')}"
    
    create_group_btn.click(gradio_create_group_wrapper, inputs=[create_group_name, create_group_desc], outputs=create_group_output)
    delete_group_btn.click(gradio_delete_group_wrapper, inputs=[delete_group_name, delete_group_force], outputs=delete_group_output)
    refresh_groups_btn.click(gradio_list_groups_wrapper, inputs=None, outputs=groups_list)
    group_info_btn.click(gradio_group_info_wrapper, inputs=[group_info_name], outputs=group_info_output)
    
    # 页面加载时刷新分组列表
    app.load(gradio_list_groups_wrapper, inputs=None, outputs=groups_list)
    
    # TTS功能连接
    tts_btn.click(
        tts_gradio_wrapper,
        inputs=[tts_text, tts_voice, tts_rate, tts_volume, tts_pitch],
        outputs=[tts_audio, tts_output]
    )

# 启动应用
if __name__ == "__main__":
    app.queue()
    
    # 创建FastAPI应用
    from fastapi import FastAPI
    
    # 添加API文档信息
    app_fastapi = FastAPI(
        title="声纹识别系统API",
        description="提供声纹注册、识别和管理功能的REST API接口",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # 先添加自定义路由，确保API接口优先注册
    add_custom_routes(app_fastapi)
    
    # 再挂载Gradio应用到/ui路径
    app_gradio = gr.mount_gradio_app(app_fastapi, app, path="/ui")
    
    # 使用uvicorn启动服务
    import uvicorn
    print("启动服务，API文档可访问: http://localhost:7860/docs 或 http://localhost:7860/redoc")
    print("Web界面可访问: http://localhost:7860/ui")
    uvicorn.run(app_fastapi, host="0.0.0.0", port=7860) 