#!/usr/bin/env bash
# ==============================================================================
# AI Services 模型下载脚本
# 支持通过 modelscope / git / curl 下载服务所需模型
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${BASE_DIR}/data"

echo "=================================================="
echo "📦 AI Services 模型下载工具"
echo "📂 目标目录: ${DATA_DIR}"
echo "=================================================="

mkdir -p "${DATA_DIR}/iic"
mkdir -p "${DATA_DIR}/openbmb"

# 1. 下载 YOLO11s (约 19MB)
YOLO_FILE="${DATA_DIR}/yolo11s.pt"
if [ -f "${YOLO_FILE}" ]; then
    echo "✅ [Vision] YOLO11s 权重已存在: ${YOLO_FILE}"
else
    echo "📥 [Vision] 正在下载 YOLO11s..."
    curl -L -o "${YOLO_FILE}" "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
    echo "✅ [Vision] YOLO11s 下载完成！"
fi

# 2. 下载 FunASR SeACo-Paraformer (约 953MB)
ASR_DIR="${DATA_DIR}/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
if [ -d "${ASR_DIR}" ] && [ "$(ls -A "${ASR_DIR}" 2>/dev/null)" ]; then
    echo "✅ [ASR] SeACo-Paraformer 已存在: ${ASR_DIR}"
else
    echo "📥 [ASR] 正在从 ModelScope 下载 SeACo-Paraformer..."
    if command -v modelscope &> /dev/null; then
        modelscope download --model iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch --local_dir "${ASR_DIR}"
    else
        git clone https://www.modelscope.cn/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git "${ASR_DIR}"
    fi
    echo "✅ [ASR] SeACo-Paraformer 下载完成！"
fi

# 3. 下载 OpenBMB VoxCPM2 (约 4.7GB)
TTS_DIR="${DATA_DIR}/openbmb/VoxCPM2"
if [ -d "${TTS_DIR}" ] && [ "$(ls -A "${TTS_DIR}" 2>/dev/null)" ]; then
    echo "✅ [TTS] VoxCPM2 已存在: ${TTS_DIR}"
else
    echo "📥 [TTS] 正在下载 OpenBMB VoxCPM2..."
    if command -v modelscope &> /dev/null; then
        modelscope download --model openbmb/VoxCPM2 --local_dir "${TTS_DIR}"
    else
        git clone https://www.modelscope.cn/openbmb/VoxCPM2.git "${TTS_DIR}"
    fi
    echo "✅ [TTS] VoxCPM2 下载完成！"
fi

# 4. 下载 BAAI bge-reranker-base (约 550MB)
RERANK_DIR="${DATA_DIR}/BAAI/bge-reranker-base"
if [ -d "${RERANK_DIR}" ] && [ "$(ls -A "${RERANK_DIR}" 2>/dev/null)" ]; then
    echo "✅ [Rerank] bge-reranker-base 已存在: ${RERANK_DIR}"
else
    echo "📥 [Rerank] 正在下载 BAAI bge-reranker-base..."
    if command -v modelscope &> /dev/null; then
        modelscope download --model BAAI/bge-reranker-base --local_dir "${RERANK_DIR}"
    else
        git clone https://www.modelscope.cn/BAAI/bge-reranker-base.git "${RERANK_DIR}"
    fi
    echo "✅ [Rerank] bge-reranker-base 下载完成！"
fi

echo "=================================================="
echo "🎉 全部模型下载与检查完毕！"
echo "=================================================="
