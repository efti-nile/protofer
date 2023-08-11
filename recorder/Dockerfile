FROM python:3.11-bookworm

RUN echo "deb http://deb.debian.org/debian buster main contrib non-free" \
    >> /etc/apt/source.list

RUN apt update && apt install -y \
    ffmpeg \
    libavcodec-extra \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev

WORKDIR /opt
RUN mkdir /opt/data
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt
COPY *.py ./

CMD ["python", "record_camera.py", "rtsp://192.168.1.88:554/11"]
