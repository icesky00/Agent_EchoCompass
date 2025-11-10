"""
Qwen 模型下载脚本
使用 ModelScope 下载 Qwen2.5 模型到本地
"""

from modelscope.hub.snapshot_download import snapshot_download
import os
import sys

def download_model(model_name, cache_dir='./qwen_models'):
    """
    下载指定的 Qwen 模型
    
    参数:
        model_name: 模型名称
        cache_dir: 缓存目录
    """
    print(f"\n{'='*60}")
    print(f"开始下载模型: {model_name}")
    print(f"缓存目录: {cache_dir}")
    print(f"{'='*60}\n")
    
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # 下载模型
        model_dir = snapshot_download(
            model_name,
            cache_dir=cache_dir,
            revision='master'
        )
        
        print(f"\n{'='*60}")
        print(f"✓ 模型下载完成！")
        print(f"✓ 模型路径: {model_dir}")
        print(f"{'='*60}\n")
        
        # 验证模型文件
        print("验证模型文件...")
        model_files = os.listdir(model_dir)
        essential_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
        
        for file in essential_files:
            if file in model_files:
                print(f"  ✓ {file}")
            else:
                print(f"  ✗ {file} (缺失)")
        
        print(f"\n模型目录包含 {len(model_files)} 个文件")
        
        return model_dir
        
    except Exception as e:
        print(f"\n❌ 下载失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════╗
║          心语罗盘 - Qwen 模型下载工具                    ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    # 可选模型配置
    models = {
        "1": {
            "name": "qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
            "desc": "7B 量化版 (推荐) - 显存 10GB",
            "size": "~7-8GB"
        },
        "2": {
            "name": "qwen/Qwen2.5-7B-Instruct",
            "desc": "7B 完整精度 - 显存 17GB",
            "size": "~14GB"
        },
        "3": {
            "name": "qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            "desc": "7B 激进量化 - 显存 6GB",
            "size": "~4GB"
        },
        "4": {
            "name": "qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
            "desc": "14B 量化版 (高精度) - 显存 18GB",
            "size": "~14GB"
        }
    }
    
    print("请选择要下载的模型：\n")
    for key, model in models.items():
        print(f"  {key}. {model['desc']}")
        print(f"     模型: {model['name']}")
        print(f"     大小: {model['size']}\n")
    
    # 用户选择
    choice = input("请输入选项 (1-4，直接回车默认选 1): ").strip() or "1"
    
    if choice not in models:
        print("❌ 无效选项")
        sys.exit(1)
    
    selected_model = models[choice]
    
    print(f"\n已选择: {selected_model['desc']}")
    print(f"预计下载大小: {selected_model['size']}")
    
    confirm = input("\n确认下载？(Y/n): ").strip().lower()
    if confirm and confirm != 'y':
        print("已取消")
        sys.exit(0)
    
    # 开始下载
    download_model(selected_model['name'])
    
    print("""
╔══════════════════════════════════════════════════════════╗
║  下载完成！现在可以启动后端服务:                         ║
║  $ python main.py                                        ║
╚══════════════════════════════════════════════════════════╝
    """)
