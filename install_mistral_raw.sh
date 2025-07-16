#!/bin/bash
cd "$(dirname "$0")"

# Written by StormTheory
# https://github.com/stormtheory/friday-ai

### Creates or opens the virtual Enviorment needed for AI tools to run
##### Note you will need at least 4G of /tmp space available for the startup install.
##### Virtual environment may take up 7Gbs of space for all needed packages.
##### Runs the creating and installing of the virtual environment setup one time.

# No running as root!
ID=$(id -u)
if [ "$ID" == '0'  ];then
        echo "Not safe to run as root... exiting..."
        exit
fi

if [ -d ./.venv ];then
### Checking dependencies
	APT_LIST=$(apt list 2>/dev/null)
	if echo "$APT_LIST"|grep -q nvidia-driver;then
                echo "✅ Installed... nvidia-driver"
                if echo "$APT_LIST"|grep -q nvidia-cuda-toolkit;then
                        echo "✅ Installed... nvidia-cuda-toolkit"
                else
                        read -p "⚠️ Install nvidia-cuda-toolkit for Image Gen? [y] > " ANS
                        if [ "$ANS" == y ];then
                                sudo apt install nvidia-cuda-toolkit
                        fi
                fi
        else
                echo "⚠️  nvidia-driver not installed!... exiting..."
                exit
        fi

#### Add to the Env Box
	# 1. Activate it
		source ./.venv/bin/activate

	# 2. Update
		pip install --upgrade pip

#### Mistral
	if [ $(which cmake | wc -l) -gt 0 ]; then
  		echo "✅ CMake found."
	else
  		echo "⚠️ CMake not found. Installing..."
  		sudo  apt install -y cmake
	fi

	# ⚠️ bitsandbytes works best with NVIDIA GPUs. For CPU-only, consider ggml or ctransformers.
	pip install torch transformers accelerate bitsandbytes fastapi uvicorn

	mkdir -p ./.venv/models/mistral_raw/
	cd -
else
	echo "⚠️  ./.venv which is the Virtual Environment is not present..."
	exit
fi
	exit
