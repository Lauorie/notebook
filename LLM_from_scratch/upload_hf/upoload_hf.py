# 上传模型到 Hugging Face
from huggingface_hub import login, HfApi
from pathlib import Path

def upload_model_to_hf(model_path, repo_id, token):
    try:
        # 验证路径是否存在
        if not Path(model_path).exists():
            raise ValueError(f"Model path does not exist: {model_path}")
            
        # 登录 Hugging Face
        login(token=token)
        print("登录成功！")
        
        # 初始化 Hugging Face API
        api = HfApi()
        
        # 上传模型（文件夹）到指定的仓库
        print(f"开始上传到仓库 {repo_id} ...")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=model_path,
            commit_message="上传本地模型",
            token=token,
            repo_type="model"    # 指定仓库类型为模型仓库
        )
        print("上传成功！")
        
    except Exception as e:
        print(f"上传过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 配置信息
    HF_TOKEN = "hf_wDVmUlCwBiymJkQVsQXermjbVKmUVyEg" # hRKF
    REPO_ID = "Laurie/Qwen2.5-7b-data-classification"
    MODEL_PATH = "./checkpoint-362"
    
    # 执行上传
    upload_model_to_hf(MODEL_PATH, REPO_ID, HF_TOKEN)