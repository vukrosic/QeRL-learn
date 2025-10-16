GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]"
pip install torch==2.7.1
pip install torchaudio==2.7.1
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install trl==0.21.0
pip install vllm==0.10.1
# replace vllm/vllm/lora/models.py with vllm_replacement/models.py
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv replacement/vllm_replacement/models.py $site_pkg_path/vllm/lora/models.py
# replace vllm/vllm/lora/worker_manager.py with vllm_replacement/worker_manager.py
cp -rv replacement/vllm_replacement/worker_manager.py $site_pkg_path/vllm/lora/worker_manager.py
# make an empty folder to pass asserts in vllm lora requests
mkdir -p simon_lora_path simon_stub_path

pip install peft

git clone --branch 0.11.0 --depth 1 https://github.com/neuralmagic/compressed-tensors.git
cd compressed-tensors
pip install -e . --no-deps
cd ..
# replace compressed-tensors/src/compressed_tensors/linear/compressed_linear.py with compressed-tensors_replacement/compressed_linear.py 
cp replacement/compressed-tensors_replacement/compressed_linear.py compressed-tensors/src/compressed_tensors/linear/compressed_linear.py
# replace compressed-tensors/src/compressed_tensors/quantization/lifecycle/forward.py with compressed-tensors_replacement/forward.py
cp replacement/compressed-tensors_replacement/forward.py compressed-tensors/src/compressed_tensors/quantization/lifecycle/forward.py

pip install accelerate==1.10.1 --no-deps

site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv replacement/trainer.py $site_pkg_path/transformers/trainer.py
