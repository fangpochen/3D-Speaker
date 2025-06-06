# TTS (文本转语音) 功能说明

## 概述

本系统集成了基于 Microsoft Edge TTS 的文本转语音功能，支持多种语言和语音类型。

## 功能特点

- 🌍 **多语言支持**: 中文、英文、日语、韩语等
- 🎤 **多种语音**: 男声、女声，不同音色
- ⚙️ **参数调节**: 语速、音量、音调可调
- 🔄 **API接口**: 支持REST API调用
- 💾 **文件管理**: 自动保存和管理生成的音频文件

## Web界面使用

### 1. 打开TTS选项卡
在Gradio界面中点击"文本转语音(TTS)"选项卡。

### 2. 输入设置
- **输入文本**: 在文本框中输入要转换的文本
- **选择语音**: 从下拉菜单选择合适的语音
- **调节参数**: 
  - 语速: -50% 到 +100%
  - 音量: -50% 到 +100%  
  - 音调: -100Hz 到 +100Hz

### 3. 生成语音
点击"生成语音"按钮，系统会处理请求并返回音频文件。

## API接口

### 1. 文本转语音

**POST** `/api/tts`

```json
{
    "text": "你好，世界！",
    "voice": "zh-CN-XiaoyiNeural",
    "rate": "+20%",
    "volume": "+10%", 
    "pitch": "+0Hz"
}
```

**响应:**
```json
{
    "success": true,
    "message": "TTS转换成功",
    "data": {
        "filename": "tts_abc123.mp3",
        "audio_path": "/path/to/audio",
        "download_url": "/api/tts/download/tts_abc123.mp3"
    }
}
```

### 2. 下载音频文件

**GET** `/api/tts/download/{filename}`

直接下载生成的音频文件。

### 3. 获取可用语音列表

**GET** `/api/tts/voices?language=zh-cn`

**响应:**
```json
{
    "success": true,
    "message": "获取到 X 个语音",
    "data": {
        "voices": [
            {
                "Name": "Microsoft Server Speech Text to Speech Voice (zh-CN, XiaoyiNeural)",
                "ShortName": "zh-CN-XiaoyiNeural",
                "Gender": "Female",
                "Locale": "zh-CN"
            }
        ]
    }
}
```

### 4. 表单格式接口

**POST** `/direct/tts` (表单数据)

```
text: 你好世界
voice: zh-CN-XiaoyiNeural
rate: +20%
volume: +10%
pitch: +0Hz
```

## 支持的语音类型

### 中文语音
- `zh-CN-XiaoyiNeural` - 晓伊(女声)
- `zh-CN-YunyangNeural` - 云扬(男声)
- `zh-CN-XiaoxiaoNeural` - 晓晓(女声)
- `zh-CN-YunxiNeural` - 云希(男声)

### 英文语音
- `en-US-AriaNeural` - Aria(女声)
- `en-US-DavisNeural` - Davis(男声)

### 其他语言
- `ja-JP-NanamiNeural` - 日语女声
- `ko-KR-SunHiNeural` - 韩语女声

## 参数说明

### 语速 (rate)
- 格式: `+20%` 或 `-10%`
- 范围: -50% 到 +100%
- 默认: `+0%`

### 音量 (volume)
- 格式: `+20%` 或 `-10%`
- 范围: -50% 到 +100%
- 默认: `+0%`

### 音调 (pitch)
- 格式: `+50Hz` 或 `-20Hz`
- 范围: -100Hz 到 +100Hz
- 默认: `+0Hz`

## 测试方法

### 1. 安装依赖
```bash
pip install edge-tts
```

### 2. 运行测试脚本
```bash
python test_tts.py
```

### 3. API测试示例

```python
import requests

# 测试TTS API
response = requests.post("http://localhost:7860/api/tts", json={
    "text": "Hello, this is a test.",
    "voice": "en-US-AriaNeural"
})

if response.status_code == 200:
    result = response.json()
    print(f"音频文件: {result['data']['filename']}")
    print(f"下载链接: {result['data']['download_url']}")
```

## 文件存储

- 生成的音频文件存储在 `voice_auth_db/tts_audio/` 目录
- 文件命名格式: `tts_{uuid}.mp3`
- 音频格式: MP3

## 注意事项

1. **网络连接**: edge-tts需要网络连接访问Microsoft服务
2. **文件清理**: 系统不会自动清理生成的音频文件，请定期清理
3. **并发限制**: 建议控制并发TTS请求数量
4. **文本长度**: 建议单次转换文本长度不超过1000字符

## 故障排除

### 1. 网络错误
```
Error: Failed to connect to TTS service
```
**解决方案**: 检查网络连接，确保可以访问Microsoft服务

### 2. 语音不存在
```
Error: Voice not found
```
**解决方案**: 使用 `/api/tts/voices` 接口获取可用语音列表

### 3. 参数格式错误
```
Error: Invalid rate format
```
**解决方案**: 确保参数格式正确，如 `+20%`、`-10Hz`

方总牛逼 
docker build   --build-arg http_proxy=http://host.docker.internal:7890   --build-arg https_proxy=http://host.docker.internal:7890   --build-arg HTTP_PROXY=http://host.docker.internal:7890   --build-arg HTTPS_PROXY=http://host.docker.internal:7890   -t voice-auth-app .
