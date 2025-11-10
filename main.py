"""
心语罗盘 - FastAPI 后端服务
基于 Qwen2.5 模型的情感分析和对话建议 API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
import asyncio
import json
import re
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== FastAPI 应用初始化 ==================
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="心语罗盘 API",
    description="基于 Qwen 的情感分析和对话建议服务",
    version="1.0.0"
)

# 新增：挂载静态文件目录（让服务器可以通过网络提供static文件夹里的文件）
app.mount("/static", StaticFiles(directory="static"), name="static")
app = FastAPI(
    title="心语罗盘 API",
    description="基于 Qwen 的情感分析和对话建议服务",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型变量
model = None
tokenizer = None

# ================== 数据模型 ==================
class EmotionAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="待分析文本")

class EmotionAnalysisResponse(BaseModel):
    emotion_type: str = Field(..., description="情感类型：快乐/悲伤/愤怒/焦虑/恐惧/中性")
    empathy_score: float = Field(..., ge=0.0, le=1.0, description="共情得分 0-1")
    interest_keywords: List[str] = Field(..., description="兴趣关键词列表")
    confidence: float = Field(..., ge=0.0, le=1.0, description="分析置信度")

class SuggestionRequest(BaseModel):
    dialogue_context: List[str] = Field(..., min_items=1, description="对话历史")
    relationship_status: str = Field(..., description="关系状态")
    user_emotion: Optional[str] = Field(None, description="用户情感状态")

class SuggestionResponse(BaseModel):
    suggestion_text: str = Field(..., description="建议回复")
    reasoning: Optional[str] = Field(None, description="建议理由")

# ================== 启动/关闭事件 ==================
@app.on_event("startup")
async def load_model():
    """应用启动时加载模型"""
    global model, tokenizer
    
    logger.info("正在加载 Qwen 模型...")
    
    # 模型配置 - 根据你的环境选择
    model_name = "./qwen_models/qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"  # 推荐：显存 10GB
    # model_name = "qwen/Qwen2.5-7B-Instruct"  # 备选：完整精度，显存 17GB
    
    try:
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=False
        )
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",      # 自动选择最佳精度
            device_map="auto",       # 自动分配 GPU
            trust_remote_code=False
        )
        
        # 性能优化（Ampere+ GPU）
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("✓ TF32 加速已启用")
        
        # 模型预热
        logger.info("正在预热模型...")
        warmup_texts = ["你好", "分析情感", "测试"]
        for text in warmup_texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                _ = model.generate(**inputs, max_new_tokens=10)
        
        torch.cuda.synchronize()
        
        logger.info(f"✓ 模型加载完成")
        logger.info(f"✓ 运行设备: {model.device}")
        logger.info(f"✓ 显存占用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

@app.on_event("shutdown")
async def cleanup():
    """应用关闭时清理资源"""
    global model, tokenizer
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("✓ 资源已清理")

# ================== 推理函数 ==================
def _generate_response(messages: List[Dict], max_tokens: int = 512) -> str:
    """同步推理函数"""
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    
    return response.strip()

def extract_json_from_text(text: str) -> dict:
    """从模型回复中提取 JSON"""
    # 方法1: 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 方法2: 提取代码块中的 JSON
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 方法3: 提取第一个 JSON 对象
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"无法从响应中提取有效 JSON: {text[:200]}")

# ================== API 端点 ==================
@app.get("/")
async def root():
    return {
        "提示": "心语罗盘 API 服务已正常启动",
        "可用接口列表": {
            "1. 情感分析": "POST /analyze_emotion（参数：{\"text\": \"你的文本\"}）",
            "2. 对话建议": "POST /generate_suggestion（参数：对话历史等）",
            "3. 健康检查": "GET /health（查看服务状态）"
        },
        "使用方法": "用Postman或代码发送POST请求到对应的接口"
    }
@app.post("/analyze_emotion", response_model=EmotionAnalysisResponse)
async def analyze_emotion(request: EmotionAnalysisRequest):
    """
    情感分析接口
    
    输入: 文本
    输出: 情感类型、共情得分、兴趣关键词、置信度
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    logger.info(f"收到情感分析请求，文本长度: {len(request.text)}")
    
    # 构建提示词
    system_prompt = """你是专业的情感分析助手。分析文本并返回 JSON 格式结果。

输出格式（严格遵守）：
{
  "emotion_type": "情感类型（快乐/悲伤/愤怒/焦虑/恐惧/中性 之一）",
  "empathy_score": 0.0到1.0的数字（表示需要多少共情关怀，负面情绪得分高）,
  "interest_keywords": ["关键词1", "关键词2"],
  "confidence": 0.0到1.0的数字（分析置信度）
}

只返回 JSON，不要有其他文字。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"分析以下文本的情感：\n\n{request.text}"}
    ]
    
    # 异步执行推理
    loop = asyncio.get_event_loop()
    try:
        response_text = await loop.run_in_executor(
            None,
            lambda: _generate_response(messages, max_tokens=256)
        )
        
        logger.info(f"模型原始输出: {response_text[:100]}...")
        
        # 解析 JSON
        result = extract_json_from_text(response_text)
        
        logger.info(f"情感分析完成: {result['emotion_type']}")
        return EmotionAnalysisResponse(**result)
    
    except ValueError as e:
        logger.error(f"JSON 解析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"JSON 解析失败: {str(e)}")
    except Exception as e:
        logger.error(f"推理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

@app.post("/generate_suggestion", response_model=SuggestionResponse)
async def generate_suggestion(request: SuggestionRequest):
    """
    对话建议接口
    
    输入: 对话上下文、关系状态、用户情感
    输出: 建议回复、理由
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    logger.info(f"收到建议生成请求，对话轮数: {len(request.dialogue_context)}")
    
    # 构建对话上下文
    context_text = "\n".join([f"- {msg}" for msg in request.dialogue_context[-5:]])
    emotion_info = f"用户情感: {request.user_emotion}\n" if request.user_emotion else ""
    
    system_prompt = f"""你是善解人意的对话顾问。基于对话历史和双方的关系状态，提供恰当的回复建议来促进双方感情升温，尽可能的有情感一些，不要有人机味道，还要符合回复人的性格特点。

关系状态: {request.relationship_status}
{emotion_info}
返回 JSON 格式：
{{
  "suggestion_text": "建议的回复内容",
  "reasoning": "为什么这样回复（简短说明）"
}}

只返回 JSON，不要有其他文字。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"对话历史:\n{context_text}\n\n请提供回复建议。"}
    ]
    
    loop = asyncio.get_event_loop()
    try:
        response_text = await loop.run_in_executor(
            None,
            lambda: _generate_response(messages, max_tokens=512)
        )
        
        logger.info(f"模型原始输出: {response_text[:100]}...")
        
        result = extract_json_from_text(response_text)
        
        logger.info(f"建议生成完成")
        return SuggestionResponse(**result)
    
    except ValueError as e:
        logger.error(f"JSON 解析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"JSON 解析失败: {str(e)}")
    except Exception as e:
        logger.error(f"推理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    }

# ================== 启动服务器 ==================
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,  # GPU 模型必须单 worker
        timeout_keep_alive=30,
    )
