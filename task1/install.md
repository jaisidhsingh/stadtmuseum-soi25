conda create -n soi_task1 python=3.10 -y 
conda activate soi_task1 

conda install -c conda-forge numpy pillow  opencv pip -y

conda install -c nvidia cuda-toolkit=12.4 cudnn=9 -y

pip install onnxruntime-gpu rembg[gpu]


test if onnx works: python -c "import onnxruntime as ort; print(ort.get_available_providers())"
Expected result: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']