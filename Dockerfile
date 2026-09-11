FROM python:3.11-slim-bookworm

# Build args so files written into mounted volumes stay owned by you, not root.
ARG UID=1000
ARG GID=1000
ARG USER=dev

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# Runtime libs for OpenCV's GUI (Qt/XCB), V4L2 cameras, video codecs and USB serial.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libsm6 libxext6 libxrender1 \
      libxcb1 libxcb-xinerama0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 \
      libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 \
      libxcb-shm0 libxcb-sync1 libxcb-util1 libxcb-xfixes0 libxcb-xkb1 \
      libxkbcommon-x11-0 libdbus-1-3 libfontconfig1 \
      libv4l-0 v4l-utils \
      libusb-1.0-0 usbutils \
      fonts-dejavu-core \
      git curl nano \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# OpenCV's bundled Qt looks for fonts in its own directory and warns on every
# window creation when they are missing. Point it at the DejaVu fonts instead.
RUN QT_DIR="$(python -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')/qt" \
 && mkdir -p "$QT_DIR/fonts" \
 && cp /usr/share/fonts/truetype/dejavu/*.ttf "$QT_DIR/fonts/"

# Non-root user matching the host UID/GID; group 20 = dialout on Debian & Ubuntu,
# which is what owns /dev/ttyUSB* for the Dobot's CP210x adapter.
RUN groupadd -g ${GID} ${USER} 2>/dev/null || true \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER} \
 && usermod -aG dialout,video ${USER}

WORKDIR /work
USER ${USER}
ENV PYTHONPATH=/work

CMD ["bash"]
