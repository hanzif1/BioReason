import os

# --- 1. 配置镜像站 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 2. 确保关闭极速下载，使用基础下载模式保证稳定性 ---
if 'HF_HUB_ENABLE_HF_TRANSFER' in os.environ:
    del os.environ['HF_HUB_ENABLE_HF_TRANSFER']

from huggingface_hub import snapshot_download, logging

# --- 3. 设置日志等级 ---
logging.set_verbosity_info()

# 配置参数
repo_id = "imageomics/bioclip-2"
local_save_path = "./bioclip-2"  # 你想保存到的本地文件夹路径

print(f"正在从镜像站下载模型: {repo_id} ...")

try:
    # 开始下载
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",         # 注意这里是 model
        local_dir=local_save_path, 
        local_dir_use_symlinks=False,
        resume_download=True,      # 断点续传
        # 如果你只想下载特定文件（例如只想要 pytorch 模型），可以取消下面这行的注释
        # allow_patterns=["*.bin", "*.json", "*.txt"], 
    )
    print(f"\n✅ 下载完成！文件已保存至: {os.path.abspath(local_save_path)}")
except Exception as e:
    print("\n❌ 下载失败。请检查网络或镜像站状态：")
    print(e)