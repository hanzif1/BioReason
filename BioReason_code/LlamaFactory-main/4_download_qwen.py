import os
from huggingface_hub import snapshot_download, logging

# --- 1. 配置镜像站（针对国内服务器优化，下载速度更快） ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

if 'HF_HUB_ENABLE_HF_TRANSFER' in os.environ:
    del os.environ['HF_HUB_ENABLE_HF_TRANSFER']
# --- 2. 设置环境变量防止权限问题 ---
# 确保缓存不会跑回 /home 目录，强制指向 data2
os.environ['HF_HOME'] = '/data1/pretrained/bioclip'

# --- 3. 设置日志 ---
logging.set_verbosity_info()

# --- 4. 配置参数 ---
# 修改为 Qwen2.5-VL 模型 ID
repo_id = "Qwen/Qwen2.5-VL-7B-Instruct"
# 修改为你希望存储模型的本地绝对路径
local_dir = "/data1/pretrained/bioclip/models/Qwen2.5-VL-7B"

print(f"🚀 开始从镜像站下载模型: {repo_id}")
print(f"📂 目标目录: {local_dir}")

# 确保目标文件夹存在
if not os.path.exists(local_dir):
    os.makedirs(local_dir, exist_ok=True)

try:
    # --- 5. 执行下载 ---
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",              # 明确指定下载的是 model
        local_dir=local_dir,
        local_dir_use_symlinks=False,   # 设为 False，直接下载真实文件到 local_dir
        resume_download=True,           # 开启断点续传，接力之前下载失败的部分
        max_workers=8                   # 开启 8 线程并行下载
    )
    print(f"\n✅ 下载成功！模型已保存至: {local_dir}")

except Exception as e:
    print("\n❌ 下载过程中出错：")
    print(e)
    print("\n💡 提示：如果遇到 'Disk quota exceeded'，请检查 /data2 的剩余空间。")
    print("命令：df -h /data2")