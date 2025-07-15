#!/bin/bash
cd "$(dirname "$0")"

# Written by StormTheory
# https://github.com/stormtheory/friday-ai

### Creates or opens the virtual Enviorment needed for AI tools to run
##### Note you will need at least 4G of /tmp space available for the startup install.
##### Virtual environment may take up 7Gbs of space for all needed packages.
##### Runs the creating and installing of the virtual environment setup one time.

if [ -d ./.venv ];then
#### Build the Env Box
	APT_LIST=$(apt list 2>/dev/null)
	if echo "$APT_LIST"| grep -q nvidia-driver;then
                if echo "$APT_LIST" |grep -q nvidia-cuda-toolkit;then
                        echo "✅ Installed..."
                else
                        read -p "⚠️ Install nvidia-cuda-toolkit for Image Gen? [y] > " ANS
                        if [ "$ANS" == y ];then
                                sudo apt install nvidia-cuda-toolkit
                        fi
                fi
        fi
exit
	
	# 1. Activate it
		source ./.venv/bin/activate

	# 2. Update
		pip install --upgrade pip

#### Image Generaters
	# For CUDA 11.8 (check your version: nvidia-smi)
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install --upgrade pip setuptools wheel
	pip install diffusers[torch] transformers[vision] accelerate safetensors
	pip install torch transformers accelerate
	pip install diffusers transformers accelerate safetensors
	pip install xformers

#### Mistral
	if [ $(which cmake | wc -l) -gt 0 ]; then
  		echo "✅ CMake found."
	else
  		echo "⚠️ CMake not found. Installing..."
  		sudo  apt install -y cmake
	fi

	pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python-cu121
	pip install git+https://github.com/huggingface/huggingface_hub.git
	#pip install huggingface-hub
	mkdir -p ./.venv/models/mistral/
	cd ./.venv/models/mistral/
	huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.1-GGUF mistral-7b-instruct-v0.1.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
	cd -
else
	echo "⚠️  ./.venv which is the Virtual Environment is not present..."
	exit
fi
	exit
