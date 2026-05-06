import os
from huggingface_hub import snapshot_download, logging

# --- 1. 配置镜像站（针对国内服务器优化） ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 2. 稳定性配置 ---
if 'HF_HUB_ENABLE_HF_TRANSFER' in os.environ:
    del os.environ['HF_HUB_ENABLE_HF_TRANSFER']

# --- 3. 设置日志 ---
logging.set_verbosity_info()

# --- 4. 配置参数 ---
# 数据集 ID
repo_id = "imageomics/IDLE-OO-Camera-Traps"
# 本地存储路径
local_dir = "/data0/data/A_bioclip/IDLE-OO-Camera-Traps"

print(f"开始从镜像站下载相机陷阱数据集: {repo_id}")
print(f"目标目录: {local_dir}")

if not os.path.exists(local_dir):
    os.makedirs(local_dir, exist_ok=True)

try:
    # --- 5. 执行下载 ---
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False, # 建议设为 False，直接下载真实文件
        resume_download=True,         # 断点续传
        max_workers=8                 # 线程数
    )
    print(f"\n✅ 下载成功！数据集已保存至: {local_dir}")

except Exception as e:
    print("\n❌ 下载过程中出错：")
    print(e)
    print("\n提示：如果连接超时，重新运行脚本即可自动断点续传。")