# AeroMind edge pipeline on Jetson (JetPack 6 / L4T r36). NOT YET VALIDATED ON HARDWARE.
FROM nvcr.io/nvidia/l4t-jetpack:r36.2.0
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip && rm -rf /var/lib/apt/lists/*
# ONNX Runtime GPU wheel for JetPack 6 from the Jetson AI Lab index (check the current URL).
RUN pip3 install --no-cache-dir onnxruntime-gpu --extra-index-url https://pypi.jetson-ai-lab.dev/jp6/cu122
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip3 install --no-cache-dir -e ".[onnx]"
VOLUME /models
CMD ["python3", "-m", "aeromind", "demo", "--backend", "onnx", "--onnx-dir", "/models", "--mode", "bearing_wear"]
