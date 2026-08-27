#!/usr/bin/env python3
"""
AI 服务模型一键下载脚本
支持下载当前服务所需的全部模型：
1. ASR 主模型: FunASR SeACo-Paraformer Large (ModelScope)
2. ASR 辅助模型: FSMN-VAD & CT-Punc (ModelScope)
3. TTS 模型: OpenBMB VoxCPM2 (ModelScope / HuggingFace)
4. Vision 模型: Ultralytics YOLO11s (GitHub Releases)
"""

import argparse
import os
import subprocess
import sys
import urllib.request

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

MODELS = {
    "asr": {
        "name": "FunASR SeACo-Paraformer Large",
        "target_dir": os.path.join(DATA_DIR, "iic", "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
        "modelscope_id": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "git_url": "https://www.modelscope.cn/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git",
    },
    "vad": {
        "name": "FunASR FSMN-VAD (语音活动检测)",
        "target_dir": os.path.join(DATA_DIR, "iic", "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
        "modelscope_id": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "git_url": "https://www.modelscope.cn/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch.git",
    },
    "punc": {
        "name": "FunASR CT-Transformer Punc (标点恢复)",
        "target_dir": os.path.join(DATA_DIR, "iic", "punc_ct-transformer_cn-en-common-vocab471067-large"),
        "modelscope_id": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        "git_url": "https://www.modelscope.cn/iic/punc_ct-transformer_cn-en-common-vocab471067-large.git",
    },
    "tts": {
        "name": "OpenBMB VoxCPM2 (TTS & 声音克隆)",
        "target_dir": os.path.join(DATA_DIR, "openbmb", "VoxCPM2"),
        "modelscope_id": "openbmb/VoxCPM2",
        "huggingface_id": "openbmb/VoxCPM2",
        "git_url_ms": "https://www.modelscope.cn/openbmb/VoxCPM2.git",
        "git_url_hf": "https://huggingface.co/openbmb/VoxCPM2",
    },
    "vision": {
        "name": "Ultralytics YOLO11s",
        "target_file": os.path.join(DATA_DIR, "yolo11s.pt"),
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
    },
    "rerank": {
        "name": "BAAI bge-reranker-base (文本重排，~550MB)",
        "target_dir": os.path.join(DATA_DIR, "BAAI", "bge-reranker-base"),
        "modelscope_id": "BAAI/bge-reranker-base",
        "huggingface_id": "BAAI/bge-reranker-base",
        "git_url_ms": "https://www.modelscope.cn/BAAI/bge-reranker-base.git",
        "git_url_hf": "https://huggingface.co/BAAI/bge-reranker-base",
    },
}


def download_file(url: str, output_path: str):
    """带进度条的文件下载"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024 * 1024:
        print(f"✅ 文件已存在，跳过下载: {output_path}")
        return

    print(f"📥 正在下载 {url} -> {output_path}...")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  [{percent:5.1f}%] {mb_downloaded:6.1f}MB / {mb_total:6.1f}MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, output_path, reporthook=_progress)
    print("\n✅ 下载完成！")


def download_repo(model_key: str, source: str = "modelscope"):
    """下载目录型模型（优先使用 SDK，回退使用 git clone）"""
    cfg = MODELS[model_key]
    target_dir = cfg["target_dir"]

    if os.path.exists(target_dir) and os.listdir(target_dir):
        print(f"✅ 模型目录已存在且非空，跳过: {target_dir}")
        return

    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    print(f"\n🚀 开始下载 [{cfg['name']}] -> {target_dir}")

    # 1. 尝试使用 modelscope SDK
    if source == "modelscope" and "modelscope_id" in cfg:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            print(f"  使用 ModelScope SDK 下载: {cfg['modelscope_id']}...")
            snapshot_download(cfg["modelscope_id"], local_dir=target_dir)
            print(f"✅ [{cfg['name']}] 下载完成！")
            return
        except ImportError:
            print("  (未安装 modelscope，尝试 git clone)")
        except Exception as e:
            print(f"  ModelScope SDK 下载报错: {e}，尝试 git clone...")

    # 2. 尝试使用 huggingface_hub SDK
    if source == "huggingface" and "huggingface_id" in cfg:
        try:
            from huggingface_hub import snapshot_download
            print(f"  使用 HuggingFace Hub SDK 下载: {cfg['huggingface_id']}...")
            snapshot_download(repo_id=cfg["huggingface_id"], local_dir=target_dir)
            print(f"✅ [{cfg['name']}] 下载完成！")
            return
        except ImportError:
            print("  (未安装 huggingface_hub，尝试 git clone)")
        except Exception as e:
            print(f"  HuggingFace SDK 下载报错: {e}，尝试 git clone...")

    # 3. 回退使用 git clone
    git_url = cfg.get("git_url") or (cfg.get("git_url_ms") if source == "modelscope" else cfg.get("git_url_hf"))
    print(f"  使用 git clone 下载: {git_url}...")
    cmd = ["git", "clone", git_url, target_dir]
    subprocess.run(cmd, check=True)
    print(f"✅ [{cfg['name']}] 下载完成！")


def main():
    parser = argparse.ArgumentParser(description="NanoSense 模型一键下载工具")
    parser.add_argument("--all", action="store_true", default=True, help="下载全部模型 (默认)")
    parser.add_argument("--asr", action="store_true", help="仅下载 ASR 识别全栈模型 (SeACo-Paraformer + VAD + Punc)")
    parser.add_argument("--tts", action="store_true", help="仅下载 TTS 语音合成与声音克隆模型 (VoxCPM2)")
    parser.add_argument("--vision", action="store_true", help="仅下载目标检测模型 (YOLO11s)")
    parser.add_argument("--rerank", action="store_true", help="仅下载文本重排模型 (bge-reranker-v2-m3)")
    parser.add_argument("--source", choices=["modelscope", "huggingface"], default="modelscope",
                        help="下载源 (国内推荐 modelscope，默认)")

    args = parser.parse_args()

    select_specific = args.asr or args.tts or args.vision or args.rerank
    download_asr = args.asr or not select_specific
    download_tts = args.tts or not select_specific
    download_vision = args.vision or not select_specific
    download_rerank = args.rerank or not select_specific

    print("=" * 65)
    print("🤖 NanoSense 模型下载管理器")
    print(f"📁 数据存放目录: {DATA_DIR}")
    print(f"🌐 默认优先镜像: {args.source}")
    print("=" * 65)

    if download_vision:
        print(f"\n🎯 目标检测模型: {MODELS['vision']['name']}")
        download_file(MODELS["vision"]["url"], MODELS["vision"]["target_file"])

    if download_asr:
        download_repo("asr", source=args.source)
        download_repo("vad", source=args.source)
        download_repo("punc", source=args.source)

    if download_tts:
        download_repo("tts", source=args.source)

    if download_rerank:
        download_repo("rerank", source=args.source)

    print("\n" + "=" * 65)
    print("🎉 选定模型已全部准备就绪！")
    print("=" * 65)


if __name__ == "__main__":
    main()
