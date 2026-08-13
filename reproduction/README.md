# 生产数据复现快照

`data/ruichuang-production-snapshot-20260813.tar.gz` 是从新版生产环境以逻辑方式导出的私有复现快照。它包含：

- PostgreSQL 14 自定义格式备份；
- 企业托管知识库文件；
- 比赛 RAG 文本、图片映射和混合检索索引；
- MinIO 知识对象的全部可读版本及逐对象清单；
- 服务器模型文件的路径、大小和 SHA-256 清单。

快照不包含 `.env`、数据库密码、API Token、MinIO/KES 密钥、证书私钥或 SSH 私钥。恢复时必须重新生成这些凭据。

## 完整性

快照归档 SHA-256：

```text
4b85d7e21fa7f7d9aecc8b697a5b37f150b16b303e3006888bdd1862c4d0f2b0
```

解压后还应验证内部逐文件校验：

```bash
tar -xzf reproduction/data/ruichuang-production-snapshot-20260813.tar.gz
cd 20260813T114434Z
sha256sum -c SHA256SUMS
```

也可以直接运行仓库内的跨平台检查脚本：

```bash
tools/verify_reproduction_snapshot.sh \
  reproduction/data/ruichuang-production-snapshot-20260813.tar.gz
```

## 数据恢复

PostgreSQL：

```bash
createdb ruichuang_phase3c
pg_restore --no-owner --no-privileges --exit-on-error \
  -d ruichuang_phase3c postgres/ruichuang_phase3c.dump
```

本地文件：

```bash
mkdir -p knowledge_store outputs
cp -a knowledge_store/. /path/to/app/knowledge_store/
cp -a rag_assets /path/to/app/outputs/rag_assets
```

对象存储的 `object_store/manifest.json` 记录对象键、版本号、是否为活动版本、大小和校验值。使用 `tools/restore_object_versions.py` 导入新建的 MinIO bucket；恢复过程不会复用生产环境密钥。

模型本体约 7.46 GiB，未重复放入 Git 历史。两个模型均来自公开的 Hugging Face 仓库：

- `Qwen/Qwen2.5-VL-3B-Instruct`，生产版本固定为提交 `66285546d2b821cf421d4f5eb2576359d3770cd3`；
- `intfloat/multilingual-e5-small`，生产环境仅保留运行所需的四个文件。

在仓库根目录执行：

```bash
python tools/download_reproduction_models.py --models-dir models
```

`inventory.json` 的 `models` 数组是生产服务器的权威逐文件清单。Qwen 下载固定版本后可逐文件比对；E5 的生产副本曾由当前 Transformers 版本重新保存，配置文件的序列化字节可能与发布仓库不同，但模型权重应以清单中的 SHA-256 为准。

生产手册引用的图片不在 `assets/manual_images`（生产环境中该目录为空）；实际图文对象已经包含于 MinIO 多版本导出和 `rag_assets` 映射，因此无需另行复制图片目录。

## 限制

该快照包含比赛数据和真实企业知识对象，只能在仓库授权成员之间使用。不得把仓库改为公开，也不得将快照转存到公开 Release、公开对象存储或公共镜像。
