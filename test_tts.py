#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS功能测试脚本
"""

import asyncio
import edge_tts
import requests
import json
from pathlib import Path

async def test_edge_tts():
    """测试基础edge-tts功能"""
    print("🔊 测试edge-tts基础功能...")
    
    try:
        # 测试文本
        text = "你好，这是一个测试语音，edge-tts功能正常工作。"
        voice = "zh-CN-XiaoyiNeural"
        
        # 创建输出目录
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # 生成语音
        communicate = edge_tts.Communicate(text, voice)
        output_file = output_dir / "test_voice.mp3"
        await communicate.save(str(output_file))
        
        print(f"✅ TTS测试成功！音频文件保存在: {output_file}")
        print(f"📝 测试文本: {text}")
        print(f"🎤 使用语音: {voice}")
        
        return True
    except Exception as e:
        print(f"❌ TTS测试失败: {e}")
        return False

async def test_voice_list():
    """测试获取语音列表"""
    print("\n📝 测试获取语音列表...")
    
    try:
        voices = await edge_tts.list_voices()
        print(f"✅ 成功获取 {len(voices)} 个语音")
        
        # 显示中文语音
        chinese_voices = [v for v in voices if "zh-CN" in v["Locale"]]
        print(f"🇨🇳 中文语音数量: {len(chinese_voices)}")
        
        for voice in chinese_voices[:5]:  # 显示前5个
            print(f"   - {voice['FriendlyName']} ({voice['ShortName']})")
        
        return True
    except Exception as e:
        print(f"❌ 获取语音列表失败: {e}")
        return False

def test_api_endpoint():
    """测试API接口"""
    print("\n🌐 测试API接口...")
    
    # 注意：这需要服务器正在运行
    base_url = "http://localhost:7860"
    
    try:
        # 测试TTS API
        tts_data = {
            "text": "Hello, this is a test for TTS API.",
            "voice": "en-US-AriaNeural",
            "rate": "+10%",
            "volume": "+0%",
            "pitch": "+0Hz"
        }
        
        response = requests.post(f"{base_url}/api/tts", json=tts_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API测试成功!")
            print(f"📂 文件名: {result['data']['filename']}")
            print(f"🔗 下载链接: {base_url}{result['data']['download_url']}")
        else:
            print(f"❌ API测试失败: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("⚠️  无法连接到服务器，请确保应用正在运行 (python voice_auth_app.py)")
    except Exception as e:
        print(f"❌ API测试出错: {e}")

async def main():
    """主测试函数"""
    print("🚀 开始TTS功能测试\n")
    
    # 测试基础TTS功能
    await test_edge_tts()
    
    # 测试语音列表
    await test_voice_list()
    
    # 测试API接口
    test_api_endpoint()
    
    print("\n✨ 测试完成!")

if __name__ == "__main__":
    asyncio.run(main()) 