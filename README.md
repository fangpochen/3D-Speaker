# 声纹识别系统

一个基于3D-Speaker模型的声纹识别应用，支持声纹注册、识别和管理功能，数据存储在MySQL数据库中。

## 功能特点

- 声纹注册：支持录制或上传音频文件注册用户声纹
- 声纹识别：根据音频识别用户身份
- 分组管理：支持多分组声纹注册和识别，不同分组下可有相同ID的用户
- 用户管理：查看和删除注册用户
- REST API支持：提供HTTP接口，便于集成到其他系统

## 快速开始

### 使用Docker运行

```bash
docker-compose up -d
```

访问 http://localhost:7860 打开声纹识别系统界面。

## 表结构

系统使用MySQL数据库存储用户和声纹信息，表结构如下：

```sql
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
);
```

## API 接口说明

系统提供以下HTTP API接口：

### 1. 用户注册

**URL**: `/direct/register_user`
**方法**: POST
**Content-Type**: multipart/form-data

**参数**:
- `data`: 音频文件 (必填)
- `user_id`: 用户ID (必填)
- `user_name`: 用户名称 (选填)
- `group_name`: 分组名称 (选填，默认为"default")

**返回示例**:
```json
{
  "success": true,
  "message": "成功注册用户 张三(ID: 001, 组: default)，声纹样本 #1"
}
```

### 2. 用户识别

**URL**: `/direct/identify_user`
**方法**: POST
**Content-Type**: multipart/form-data

**参数**:
- `data`: 音频文件 (必填)
- `threshold`: 匹配阈值，范围0.5-0.95 (选填，默认0.7)
- `group_name`: 分组名称 (选填，默认为"default")

**返回示例**:
```json
{
  "success": true,
  "message": "✅ 匹配成功! 用户: 张三 (ID: 001, 组: default)\n相似度分数: 0.8523"
}
```

### 3. 列出用户

**URL**: `/direct/list_users`
**方法**: POST
**Content-Type**: multipart/form-data

**参数**:
- `group_name`: 分组名称 (选填，默认为"default")

**返回示例**:
```json
{
  "success": true,
  "message": "分组 'default' 已注册用户列表:\n- 张三 (ID: 001), 样本数: 2\n- 李四 (ID: 002), 样本数: 1"
}
```

### 4. 删除用户

**URL**: `/direct/delete_user`
**方法**: POST
**Content-Type**: multipart/form-data

**参数**:
- `user_id`: 用户ID (必填)
- `group_name`: 分组名称 (选填，默认为"default")

**返回示例**:
```json
{
  "success": true,
  "message": "已删除用户 张三 (ID: 001, 组: default) 及其所有声纹样本"
}
```

## Java客户端调用示例

```java
import java.io.File;
import java.io.IOException;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class VoiceAuthClient {
    private static final String BASE_URL = "http://localhost:7860";
    private static final OkHttpClient client = new OkHttpClient();

    // 注册用户
    public static String registerUser(File audioFile, String userId, String userName, String groupName) throws IOException {
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("data", audioFile.getName(),
                        RequestBody.create(MediaType.parse("audio/wav"), audioFile))
                .addFormDataPart("user_id", userId)
                .addFormDataPart("user_name", userName)
                .addFormDataPart("group_name", groupName)
                .build();

        Request request = new Request.Builder()
                .url(BASE_URL + "/direct/register_user")
                .post(requestBody)
                .build();

        try (Response response = client.newCall(request).execute()) {
            return response.body().string();
        }
    }

    // 识别用户
    public static String identifyUser(File audioFile, double threshold, String groupName) throws IOException {
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("data", audioFile.getName(),
                        RequestBody.create(MediaType.parse("audio/wav"), audioFile))
                .addFormDataPart("threshold", String.valueOf(threshold))
                .addFormDataPart("group_name", groupName)
                .build();

        Request request = new Request.Builder()
                .url(BASE_URL + "/direct/identify_user")
                .post(requestBody)
                .build();

        try (Response response = client.newCall(request).execute()) {
            return response.body().string();
        }
    }

    // 列出用户
    public static String listUsers(String groupName) throws IOException {
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("group_name", groupName)
                .build();

        Request request = new Request.Builder()
                .url(BASE_URL + "/direct/list_users")
                .post(requestBody)
                .build();

        try (Response response = client.newCall(request).execute()) {
            return response.body().string();
        }
    }

    // 删除用户
    public static String deleteUser(String userId, String groupName) throws IOException {
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("user_id", userId)
                .addFormDataPart("group_name", groupName)
                .build();

        Request request = new Request.Builder()
                .url(BASE_URL + "/direct/delete_user")
                .post(requestBody)
                .build();

        try (Response response = client.newCall(request).execute()) {
            return response.body().string();
        }
    }

    // 使用示例
    public static void main(String[] args) {
        try {
            // 注册用户
            File audioFile = new File("path/to/audio.wav");
            System.out.println(registerUser(audioFile, "001", "张三", "default"));
            
            // 列出用户
            System.out.println(listUsers("default"));
            
            // 识别用户
            System.out.println(identifyUser(audioFile, 0.7, "default"));
            
            // 删除用户
            System.out.println(deleteUser("001", "default"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```

## 技术栈

- **后端**: Python, FastAPI, Gradio
- **数据库**: MySQL
- **模型**: 3D-Speaker ERes2NetV2 声纹识别模型
- **容器化**: Docker

## 参考
本项目使用了3D-Speaker开源工具包中的模型和部分代码。详情请参考: 
[3D-Speaker GitHub](https://github.com/modelscope/3D-Speaker)

## 授权协议
Apache License 2.0
