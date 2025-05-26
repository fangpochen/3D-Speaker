# 声纹识别系统 Docker 部署说明

本文档介绍如何使用 Docker 部署声纹识别系统。

## 前提条件

- 安装 [Docker](https://docs.docker.com/get-docker/)
- 安装 [Docker Compose](https://docs.docker.com/compose/install/)
- **确保 `speakerlab` 目录与 `Dockerfile` 在同一项目根目录下。** 这个目录是 3D-Speaker 工具包的一部分，包含了运行应用所需的声纹处理模块。

## 部署步骤

### 1. 下载声纹识别模型

在构建Docker镜像前，需要先下载模型文件：

```bash
# 安装modelscope
pip install modelscope

# 运行下载脚本
python download_model.py
```

脚本会将模型下载到`./modelscope_hub_cache`目录下。

### 2. 构建并启动容器

使用Docker Compose构建并启动容器：

```bash
# 构建镜像（如果修改了代码需要重新构建）
docker-compose build --no-cache

# 启动服务
docker-compose up -d
```

服务启动后，访问 http://localhost:7860 打开声纹识别系统界面。

### 3. 查看日志

```bash
docker-compose logs -f
```

### 4. 停止服务

```bash
docker-compose down
```

## 数据持久化

系统使用两个持久化目录：

1. **模型文件**：`./modelscope_hub_cache` 映射到容器内的 `/app/modelscope_hub_cache`
2. **声纹数据**：`./voice_auth_db` 映射到容器内的 `/app/voice_auth_db`

这些目录通过Docker卷挂载到容器中，确保数据持久化。

## MySQL 配置

系统使用外部 MySQL 数据库存储用户和声纹信息。数据库连接配置通过环境变量设置：

```yaml
environment:
  - MYSQL_HOST=106.13.0.50
  - MYSQL_PORT=3306
  - MYSQL_USER=root
  - MYSQL_PASSWORD=AaBb123456!
  - MYSQL_DATABASE=dcc_dbms
```

可以在 `docker-compose.yml` 文件中根据实际情况修改这些配置。

## 自定义构建选项

如果需要使用不同的模型或配置，可以修改以下文件：

1. `voice_auth_app.py`：应用主程序，包含模型ID和数据库配置
2. `download_model.py`：修改 MODEL_ID 和 MODEL_REVISION 变量
3. `docker-compose.yml`：修改端口映射、环境变量等
4. `Dockerfile`：修改基础镜像、依赖安装等

## 常见问题

### 1. 模型下载失败

如果遇到SSL证书问题导致模型下载失败，可以尝试：

```python
# 在下载脚本中添加
os.environ["SSL_CERT_FILE"] = ""
os.environ["HTTPX_SSL_VERIFY"] = "0"
```

### 2. 数据库连接问题

确保MySQL服务器允许远程连接，并且防火墙已开放3306端口。可以先在服务器上使用命令行工具测试连接：

```bash
mysql -h 106.13.0.50 -u root -p -P 3306
```

### 3. 容器内外文件权限问题

如果遇到容器内无法写入挂载目录的问题，可能是权限问题，尝试调整宿主机目录权限：

```bash
chmod -R 777 ./voice_auth_db
``` 
 docker build   --build-arg http_proxy=http://172.17.0.1:7890   --build-arg https_proxy=http://172.17.0.1:7890   --build-arg HTTP_PROXY=http://172.17.0.1:7890   --build-arg HTTPS_PROXY=http://172.17.0.1:7890   -t voice-auth-app .
