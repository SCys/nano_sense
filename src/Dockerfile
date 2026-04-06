FROM python:3.11-slim AS deps-builder

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

RUN sed -i 's/deb.debian.org/mirrors.cernet.edu.cn/g' /etc/apt/sources.list.d/debian.sources

WORKDIR /build

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir uv --index https://mirrors.cernet.edu.cn/pypi/web/simple

COPY ./pyproject.toml ./uv.lock ./

# 只导出业务依赖，剔除 runtime 基础镜像已提供的 torch 生态包。
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --no-dev --format requirements-txt --no-hashes \
    --prune torch --prune torchvision --prune torchaudio \
    -o requirements.txt


FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

# chinese mirror
RUN \
  sed -i 's/archive.ubuntu.com/mirrors.cernet.edu.cn/g' /etc/apt/sources.list && \
  sed -i 's/security.ubuntu.com/mirrors.cernet.edu.cn/g' /etc/apt/sources.list

RUN --mount=type=cache,target=/var/cache/apt \
  apt-get update && \
  apt-get install -y build-essential ffmpeg libglib2.0-0 libgl1 && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app
EXPOSE 8000

# 校验 runtime Python 版本与 pyproject 中的 ==3.11.* 对齐
RUN python --version

COPY --from=deps-builder /build/requirements.txt ./requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt \
    --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
    --extra-index-url https://pypi.org/simple

COPY . .

CMD [ "python", "main.py" ]
