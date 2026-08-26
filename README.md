# NanoSense

> **NanoSense**: 极轻量、双卡协同的高性能 AI 多模态中台服务，提供高精度语音识别（ASR）、高保真语音合成与声音克隆（TTS）、图像目标检测（Vision）与文本嵌入（Embeddings）能力。

系统采用**双卡解耦架构**与**全生命周期显存自管理机制（30分钟无请求自动释放）**，单服务即可高效承载五大 AI 能力。

---

## 🌟 核心功能一览

| 业务板块 | API 端点 | 核心模型与能力 | 运行设备 | 典型应用场景 |
|---|---|---|---|---|
| 🎙️ **语音识别 (ASR)** | `POST /v1/audio/transcriptions` | **FunASR SeACo-Paraformer**<br>• 中文/方言/中英混输高准确率<br>• `fsmn-vad` 智能切片（支持数小时长音频）<br>• `ct-punc` 智能恢复标点断句 | **GTX 1050 Ti** (`cuda:1`) | 会议纪要转写、字幕生成、语音输入法 |
| 🔊 **语音合成 (TTS)** | `POST /v1/audio/speech`<br>`POST /v1/audio/synthesize` | **OpenBMB VoxCPM2** (20亿参数)<br>• **OpenAI 兼容接口**（11 种预设音色全映射）<br>• **自然语言音色设计**（任意文字描述音色）<br>• **情绪/语气叠加**（轻快、沉稳、激动等）<br>• 原生 **48kHz 录音室级高清音质** | **Tesla T10** (`cuda:0`) | 智能助理播报、有声书朗读、数字人配音 |
| 🧬 **声音克隆 (Cloning)** | `POST /v1/audio/clone`<br>*(或 `/v1/audio/speech` 传 Base64)* | **VoxCPM2 复合克隆引擎**<br>• **3~10 秒极速音色复刻**<br>• **克隆+情绪微调**（复刻音色同时叠加语气）<br>• **极清克隆**（配合原台词极致保真） | **Tesla T10** (`cuda:0`) | 专属声音定制、角色声音复刻、个性化配音 |
| 👁️ **目标检测 (Vision)** | `POST /v1/vision/detection` | **Ultralytics YOLO11s**<br>• 毫秒级快速图像检测<br>• 输出精确边界框 (Bounding Box) 与置信度 | **Tesla T10** (`cuda:0`) | 人体/人脸识别、安防监控、画面内容分析 |
| 📝 **文本嵌入 (Embeddings)** | `POST /v1/embeddings` | **OpenAI 兼容 Embeddings**<br>• 支持单条/批量文本向量化<br>• 适配知识库召回检索 | 远程网关 | RAG 知识库检索、语义搜索、文本聚类 |

---

## 🖥️ 硬件架构与双卡分配（实测数据）

服务根据算力与显存特征，将计算负载自动分配至双卡，避免单一 GPU 显存过载：

```
                    ┌─────────────────── FastAPI / Docker (端口 5015) ───────────────────┐
                    │                                                                       │
                    ▼                                                                       ▼
      【GTX 1050 Ti (4GB)】 (cuda:1)                                          【Tesla T10 (16GB)】 (cuda:0)
  ┌─────────────────────────────────────┐                                 ┌─────────────────────────────────────┐
  │  🎙️ FunASR 全栈语音识别            │                                 │  🔊 VoxCPM2 语音合成与声音克隆      │
  │  • Paraformer + VAD + CT-Punc       │                                 │  • 20亿参数，48kHz 采样率           │
  │  • 显存占用: ~2.19 GB (余量 1.9GB)  │                                 │  • 显存占用: ~5.2 GB (余量 10.8GB)  │
  │  • RTF: 0.329 (比实时快 3 倍)       │                                 │  ─────────────────────────────────  │
  │                                     │                                 │  👁️ YOLO11s 视觉检测 (占 ~25MB)    │
  └─────────────────────────────────────┘                                 └─────────────────────────────────────┘
```

### ⚡ 30 分钟无请求自动下线机制 (Auto Offload)
* **懒加载 (Lazy Load)**：服务冷启动时 **0 MB 显存占用**，各模型仅在收到第一笔对应请求时初始化载入。
* **空闲自动卸载**：后台守护线程每 60 秒巡检一次，连续 30 分钟无请求的模型自动执行 `gc.collect()` 与 `cuda.empty_cache()`，**将占用的显存彻底归还给系统（降至 0 MB）**，下次调用时自动秒级拉起。

---

## 📦 模型下载链接与一键下载脚本

服务启动前需确保模型文件已放置在 `data/` 目录下。本项目提供了自动下载脚本，支持 ModelScope（国内高速）与 HuggingFace 源：

### 1. 一键脚本下载（推荐）

```bash
# 执行 Bash 脚本自动下载全部模型
./scripts/download_models.sh

# 或者使用 Python 脚本（支持局部下载参数：--asr / --tts / --vision）
python scripts/download_models.py --source modelscope
```

### 2. 官方模型下载链接对照表

| 模型用途 | 选定模型 | 官方仓库地址与下载方式 | 本地存放目录 |
|---|---|---|---|
| 🎙️ **ASR 识别** | FunASR SeACo-Paraformer Large | [ModelScope iic/speech_seaco_paraformer...](https://www.modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)<br>`modelscope download --model iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --local_dir data/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | `data/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| 🔊 **TTS 合成/克隆** | OpenBMB VoxCPM2 (2B) | [ModelScope openbmb/VoxCPM2](https://www.modelscope.cn/models/openbmb/VoxCPM2) / [HuggingFace openbmb/VoxCPM2](https://huggingface.co/openbmb/VoxCPM2)<br>`modelscope download --model openbmb/VoxCPM2 --local_dir data/openbmb/VoxCPM2` | `data/openbmb/VoxCPM2` |
| 👁️ **目标检测** | Ultralytics YOLO11s | [GitHub Releases v8.3.0](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)<br>`curl -L -o data/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt` | `data/yolo11s.pt` |

---

## 🚀 快速启动

### 方式一：Docker Compose（推荐，开箱即用）

```bash
cd /data
docker compose up -d --build ai
# 服务已启动，监听宿主机 http://localhost:5015
```

### 方式二：本地虚拟环境运行

```bash
cd src
uv sync --extra torch
uvicorn main:app --host 0.0.0.0 --port 8000
```

配置位于 `main.toml`（支持环境变量覆盖：`ASR_DEVICE`、`TTS_DEVICE`、`ASR_MODEL_PATH`、`TTS_MODEL_PATH`、`OPENAI_API_KEY` 等）。

---

## 📡 接口使用指南与完整 cURL 示例

### 1. 语音合成与自然语言音色设计 (`POST /v1/audio/speech`)

**OpenAI 兼容格式**，支持 11 种预设音色、自由自然语言描述音色及情绪微调：

```bash
# 示例：使用 OpenAI 预设音色 nova + 情绪控制
curl http://localhost:5015/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm2",
    "input": "您好，欢迎使用复合语音合成服务！",
    "voice": "nova",
    "instructions": "稍微放慢语速，带有一点微笑感",
    "response_format": "mp3",
    "speed": 1.0
  }' --output speech.mp3
```

```bash
# 示例：直接用自然语言描述定制任意音色
curl http://localhost:5015/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "各位听众朋友大家好，欢迎收听今晚的夜间新闻。",
    "voice": "低沉富有磁性的中年男播音员声音",
    "instructions": "沉稳专业、富有讲述感",
    "response_format": "wav"
  }' --output news.wav
```

* **预设音色列表**：`alloy`、`echo`、`fable`、`onyx`、`nova`、`shimmer`、`ash`、`ballad`、`coral`、`sage`、`verse`
* **支持格式**：`mp3`、`wav`、`flac`、`ogg`、`opus`、`aac`、`pcm`（48kHz 采样率）

---

### 2. 声音克隆 (Voice Cloning)

**方式 A：直接上传 3~10 秒录音文件复刻声音（推荐）**

```bash
curl http://localhost:5015/v1/audio/clone \
  -F "file=@speaker.wav" \
  -F "text=这是一段通过参考音频直接克隆生成的语音，音色高度还原。" \
  -F "instructions=自信沉稳的语气" \
  -F "response_format=wav" \
  --output cloned.wav
```

**方式 B：通过 OpenAI 格式传 Base64 音频**

```bash
curl http://localhost:5015/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "这是一段基于 Base64 音频克隆生成的内容。",
    "reference_audio": "data:audio/wav;base64,UklGRi...",
    "instructions": "开心的语气",
    "response_format": "mp3"
  }' --output cloned.mp3
```

* **极清克隆 (Ultimate Cloning)**：传入 `prompt_text`（参考音频对应的原文本台词），保真度最高。

---

### 3. 语音识别 (ASR) (`POST /v1/audio/transcriptions`)

支持单句短语音、会议录音、中英混合、四川/河南等方言，自动分段加标点：

```bash
# 1. 默认 JSON 格式返回
curl http://localhost:5015/v1/audio/transcriptions \
  -F "file=@meeting.mp3"

# 2. 纯文本格式直接返回
curl "http://localhost:5015/v1/audio/transcriptions?response_format=text" \
  -F "file=@voice.wav"

# 3. 详细模式 (带耗时与分段信息)
curl "http://localhost:5015/v1/audio/transcriptions?response_format=verbose_json" \
  -F "file=@interview.ogg"
```

---

### 4. 图像目标检测 (`POST /v1/vision/detection`)

```bash
curl http://localhost:5015/v1/vision/detection \
  -F "image=@photo.jpg"

# 返回示例:
# {
#   "predictions": [
#     {"x_min": 120.5, "y_min": 85.0, "x_max": 310.2, "y_max": 450.0, "confidence": 0.94, "label": "person"}
#   ]
# }
```

---

### 5. 文本嵌入 (`POST /v1/embeddings`)

```bash
curl http://localhost:5015/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "用于知识库检索与语义相似度计算的测试文本",
    "model": "text-embedding-004"
  }'
```

---

## 📊 模型选型与评测记录（重要：勿重复评估）

> 本节记录已完成的模型选型结论与实测对比。未来更换/新增模型前请先阅读，避免重新下载已被否决的候选。

### 1. TTS 选型 —— 选定 `OpenBMB VoxCPM2`（2026-08）

**候选与能力实测对比：**

| 候选模型 | 参考音频克隆 | 自然语言指定音色 | 情绪/语气复合控制 | 输出采样率 | 显存占用 | 结论 |
|---|---|---|---|---|---|---|
| ✅ **VoxCPM2** (`data/openbmb/`) | ✅ **完全支持**（3秒极速克隆） | ✅ **完全支持**（前置提示词） | ✅ **完全支持**（克隆+语气叠加） | **48kHz**（内置 AudioVAE） | ~5.2 GB (bfloat16) | **最优选，全面胜出** |
| ❌ Qwen3-TTS-VoiceDesign (`data/wavekat/`) | ❌ 不支持（无音频编码器） | ✅ 支持 | ❌ 不支持克隆叠加 | 24kHz | ~4 GB (ONNX) | 否决：无克隆能力，且 ONNX 存在 CUDA 版本依赖 |
| ❌ VibeVoice-1.5B (`data/microsoft/`) | ❌ 仅支持预设角色 | ❌ | ❌ | 24kHz | ~5.1 GB | 否决：专用于 90 分钟 4 人播客长对话，不适合常规 TTS |
| ❌ VoxCPM2-4bit (MLX) (`data/mlx-community/`) | — | — | — | — | — | 否决：为 Apple Silicon (Mac) 专有 MLX 格式，Linux 无法运行 |

**选定与清理记录（2026-08）：**
* **选定理由**：VoxCPM2 是唯一能够同时满足“自然语言音色设计” + “参考音频声音克隆” + “克隆叠加情绪控制” + “48kHz 超高音质”的全能型模型。
* **已清理的无用 TTS 模型（释放约 25.6 GB 磁盘空间）**：
  * 删除 `data/wavekat/`（Qwen3-TTS，18 GB）
  * 删除 `data/microsoft/`（VibeVoice-1.5B，5.1 GB）
  * 删除 `data/mlx-community/`（VoxCPM2-4bit MLX，2.5 GB）
  * 删除 `data/voxcpm2-4bit/`（空目录）

---

### 2. ASR 选型 —— 选定 `seaco_paraformer_large`（2026-08）

**候选与实测对比（同批 4 条中文/方言音频共 31s，预热后稳态）：**

| 候选模型 | 中文准确率 | 中英混输 | 原生标点断句 | 仅 ASR (RTF) | ASR+VAD+标点 (RTF) | 显存增量 | 结论 |
|---|---|---|---|---|---|---|---|
| ✅ **seaco_paraformer_large** (`data/iic/`) | 好（川话/河南话准确） | ✅ 正确 | ❌ 需外挂 ct-punc | **0.134** | **0.329** | ~2.04 GB | **最优选**（准确率最高，挂载标点后自然） |
| ❌ FireRedASR2-CTC int8 (`data/fire-red/`) | 好 | ❌ Monday→"MOAY TOAY" | ❌ | 0.52 (CPU) | — | — | 否决：中英混输有错字，CTC-only 裁剪版丢失解码器 |
| ❌ whisper-small fp16 (`data/whisper/`) | 差：“锅意想”“优寿”“六岗”等错字多 | 一般 | ✅ 有 | 0.120 | — | ~0.5 GB | 否决：中文错字率不可接受 |

**显存占用实测（GTX 1050 Ti，进程级）：**
- `seaco_paraformer` 主模型：~956 MB
- `fsmn-vad` 切分模型：~1 MB
- `ct-punc` 标点模型：~1,085 MB
- **全栈合计**：**~2,042 MB 分配 / 2,190 MB 预留**（稳态 RSS 约 3.8 GB，1050 Ti 剩余 ~1.9 GB 裕量）

**已清理的无用 ASR 模型（2026-08）：**
- 删除 `data/fire-red/`（FireRedASR2，742 MB）
- 删除 `data/whisper/`（Whisper 缓存 + small.pt，462 MB）

---

### 3. 目标检测 (Vision) 选型
* 选定 **`YOLO11s`**（19 MB，显存占用约 25 MB，COCO mAP 47.0%）。
* **选定理由**：在 GPU 服务器上相比 Nano 版本（39.5% mAP）带来大幅精度提升（+7.5 mAP），单帧延迟依然稳定在 18~19ms（50+ FPS），是性价比最高的中台视觉模型。
* **已清理的冗余视觉权重（2026-08）**：
  * 删除 `yolo11n.pt`、`yolo26n.pt`、`yolo26s.pt`、`yoloe-11s-seg.pt`、`yoloe-11s-seg-pf.pt`、`mobile_sam.pt`。
