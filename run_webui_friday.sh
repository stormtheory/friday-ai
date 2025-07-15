#!/bin/bash
cd "$(dirname "$0")"

# Written by StormTheory
# https://github.com/stormtheory/friday-ai

### Creates or opens the virtual Enviorment needed for AI tools to run
##### Note you will need at least 4G of /tmp space available for the startup install.
##### Virtual environment may take up 7Gbs of space for all needed packages.
##### Runs the creating and installing of the virtual environment setup one time.

if [ ! -d ./.venv ];then
#### Build the Env Box
	APT_LIST=$(apt list 2>/dev/null)
        if echo "$APT_LIST"|grep -q python3.12-venv;then
		echo "✅ Installed..."
	else
		echo "⚠️ Installing python3.12-venv"
		sudo apt install python3.12-venv
	fi
	
	if echo "$APT_LIST"|grep -q portaudio19-dev;then
		echo "✅ Installed..."
	else
	read -p "⚠️ Install portaudio19-dev for audio? [y] > " ANS
		if [ "$ANS" == y ];then
			sudo apt install portaudio19-dev
		fi
	fi

	if echo "$APT_LIST"|grep -q nvidia-driver;then
                if echo "$APT_LIST"|grep -q nvidia-cuda-toolkit;then
                        echo "✅ Installed..."
                else
                        read -p "⚠️ Install nvidia-cuda-toolkit for Image Gen? [y] > " ANS
                        if [ "$ANS" == y ];then
                                sudo apt install nvidia-cuda-toolkit
                        fi
                fi
        fi
	
	# 1. Create a virtual environment
		python3 -m venv ./.venv

	# 2. Activate it
		source ./.venv/bin/activate

	# 3. Update
		pip install --upgrade pip

#### Audio/Voice
	pip install pyttsx3

#### Voice
	if echo "$APT_LIST"|grep -q portaudio19-dev;then
		pip install SpeechRecognition pyaudio
		pip install gTTS
	fi
	
#### Image Generaters
	# For CUDA 11.8 (check your version: nvidia-smi)
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install --upgrade pip setuptools wheel
	pip install diffusers[torch] transformers[vision] accelerate safetensors
	pip install torch transformers accelerate
	pip install diffusers transformers accelerate safetensors
	pip install xformers

#### webui
	pip install gradio  # WebGUI
	pip install pymupdf # PDF

#### Indexing / RAG
	pip install sentence_transformers
	pip install langchain
	pip install sentence_transformers

	pip install faiss-cpu
	#pip install faiss-gpu
fi

#### Run the Box
	source ./.venv/bin/activate
	export PYTHONWARNINGS="ignore"
#### Run the AI
	echo "Starting the AI"
	python -m webui
	exit
