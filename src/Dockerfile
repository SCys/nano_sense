# FROM iscys.com/pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime
# FROM iscys.com/pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive PIP_PREFER_BINARY=1

# chinese mirror
RUN \
  sed -i 's/archive.ubuntu.com/mirrors.cernet.edu.cn/g' /etc/apt/sources.list && \
  sed -i 's/security.ubuntu.com/mirrors.cernet.edu.cn/g' /etc/apt/sources.list

RUN --mount=type=cache,target=/var/cache/apt \
  apt-get update && \
  apt-get install -y ffmpeg libglib2.0-0 libgl1

WORKDIR /app
VOLUME /app

EXPOSE 8000

COPY ./requirements.txt .

RUN pip config set global.index-url https://mirrors.cernet.edu.cn/pypi/web/simple

RUN pip install -r requirements.txt

CMD [ "python", "main.py" ]