# 声纹识别系统 API 文档

## 基本信息

- 基础URL: `http://[服务器IP]:7860`
- 请求/响应内容类型: 主要为 `multipart/form-data` 用于请求，`application/json` 用于响应。

## 接口说明

本系统后端使用 FastAPI 构建，提供以下 `/direct/` 路径下的API接口，推荐直接调用这些接口以获得最佳性能和兼容性。

## 1. 用户注册 API

注册新用户并保存其声纹特征。

### 请求
- 路径: `/direct/register_user`
- 方法: `POST`
- 内容类型: `multipart/form-data`
- 参数:
    - `data`: (文件类型) 音频文件 (例如: `.wav`格式)。
    - `user_id`: (文本类型) 用户ID (必填)。
    - `user_name`: (文本类型) 用户姓名 (选填, 如果不提供，则默认为用户ID)。

### 响应 (application/json)

成功时:
```json
{
  "success": true,
  "message": "成功注册用户 [user_name](ID: [user_id])，声纹样本 #[sample_count]"
}
```
失败时 (示例):
```json
{
  "success": false,
  "error": "音频处理失败，请检查格式"
}
```
或
```json
{
  "success": false,
  "error": "请提供音频文件和用户ID"
}
```

## 2. 列出用户 API

获取系统中所有已注册用户的列表。

### 请求
- 路径: `/direct/list_users`
- 方法: `POST`
- 内容类型: `application/x-www-form-urlencoded` 或 `multipart/form-data` (此接口无实际表单参数，通常客户端会自动处理)
- 参数: 无实际表单参数。

### 响应 (application/json)

成功时:
```json
{
  "success": true,
  "message": "已注册用户列表:\n- [user_name1] (ID: [user_id1]), 样本数: [count1]\n- [user_name2] (ID: [user_id2]), 样本数: [count2]"
}
```
数据库为空时:
```json
{
  "success": true,
  "message": "数据库为空，请先注册用户"
}
```

## 3. 声纹识别 API

识别用户的声纹并返回匹配结果。

### 请求
- 路径: `/direct/identify_user`
- 方法: `POST`
- 内容类型: `multipart/form-data`
- 参数:
    - `data`: (文件类型) 音频文件 (例如: `.wav`格式)。
    - `threshold`: (数值类型) 匹配阈值 (选填, 默认0.7, 范围 0.5-0.95)。

### 响应 (application/json)

成功匹配时:
```json
{
  "success": true,
  "message": "✅ 匹配成功! 用户: [user_name] (ID: [user_id])\n相似度分数: [score]"
}
```

未匹配时:
```json
{
  "success": true,
  "message": "❌ 未匹配到任何用户\n最高相似度分数: [max_score] (阈值: [threshold])"
}
```
失败时 (示例):
```json
{
  "success": false,
  "error": "音频处理失败，请检查格式"
}
```

## 4. 删除用户 API

删除特定用户及其所有声纹样本。

### 请求
- 路径: `/direct/delete_user`
- 方法: `POST`
- 内容类型: `application/x-www-form-urlencoded` 或 `multipart/form-data`
- 参数:
    - `user_id`: (文本类型) 要删除的用户ID。

### 响应 (application/json)

成功时:
```json
{
  "success": true,
  "message": "已删除用户 [user_name] (ID: [user_id]) 及其所有声纹样本"
}
```
失败时 (用户ID不存在):
```json
{
  "success": true, 
  "message": "用户ID不存在" 
}
```
注意：此处的`success: true`但message指示操作未完成是当前代码实现的一个特点。理想情况下，当用户不存在时，`success`应为`false`。

## 错误处理

所有API在发生服务器内部错误时，可能返回类似以下的JSON响应:
```json
{
  "success": false,
  "error": "<具体的错误信息描述>"
}
```
HTTP状态码通常为 `500`。
对于特定业务逻辑错误（如参数缺失、用户不存在等），`success` 可能为 `true` 但 `message` 字段会包含错误或提示信息，或者 `success` 为 `false` 且 `error` 字段包含错误信息。请仔细检查响应内容。

## 实际应用示例 (curl)

### 用户注册请求 (使用 /direct/ 路径)

```bash
curl --location --request POST 'http://[服务器IP]:7860/direct/register_user' \
--header 'User-Agent: Apifox/1.0.0 (https://apifox.com)' \
--form 'data=@"/path/to/your/audio.wav"' \
--form 'user_id="000011"' \
--form 'user_name="测试用户"'
```

### 用户识别请求 (使用 /direct/ 路径)

```bash
curl --location --request POST 'http://[服务器IP]:7860/direct/identify_user' \
--header 'User-Agent: Apifox/1.0.0 (https://apifox.com)' \
--form 'data=@"/path/to/your/audio.wav"' \
--form 'threshold="0.75"'
```

### 列出用户请求 (使用 /direct/ 路径)
```bash
curl --location --request POST 'http://[服务器IP]:7860/direct/list_users'
```

### 删除用户请求 (使用 /direct/ 路径)
```bash
curl --location --request POST 'http://[服务器IP]:7860/direct/delete_user' \
--form 'user_id="000011"'
```

## 注意事项

1.  对于文件上传的接口 (`/direct/register_user`, `/direct/identify_user`)，请确保使用 `multipart/form-data` 格式发送请求。
2.  音频文件会被自动重采样至16kHz，单声道。确保您的输入音频与此兼容或可以被成功转换。
3.  用户ID必须唯一，以避免冲突。
4.  声纹匹配阈值范围为0.5-0.95，值越高匹配要求越严格。
5.  请将 `[服务器IP]` 和 `"/path/to/your/audio.wav"` 替换为实际的服务器IP地址和音频文件路径。 