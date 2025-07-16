#!/bin/bash
cd "$(dirname "$0")"

# Written by StormTheory
# https://github.com/stormtheory/friday-ai

### Creates or opens the virtual Enviorment needed for AI tools to run
##### Note you will need at least 4G of /tmp space available for the startup install.
##### Virtual environment may take up 7Gbs of space for all needed packages.
##### Runs the creating and installing of the virtual environment setup one time.

RUN='.run_webui_installed'

# No running as root!
ID=$(id -u)
if [ "$ID" == '0'  ];then
        echo "Not safe to run as root... exiting..."
        exit
fi

if [ ! -d ./.venv ];then
        APT_LIST=$(apt list 2>/dev/null)
        ENV_INSTALL=True
        PIP_INSTALL=True
elif [ ! -f ./.venv/.$RUN ];then
        APT_LIST=$(apt list 2>/dev/null)
        ENV_INSTALL=False
        PIP_INSTALL=True
elif [ -f ./.venv/.$RUN ];then
        ENV_INSTALL=False
        PIP_INSTALL=False
else
        exit
fi

if [ "$ENV_INSTALL" == 'True' ];then
### Checking dependencies
        
        if echo "$APT_LIST"|grep -q python3.12-dev;then
                echo "✅ Installed... python3.12-dev"
        else
                echo "⚠️ Installing python3.12-dev"
                sudo apt install python3.12-dev
        fi

        if echo "$APT_LIST"|grep -q python3.12-venv;then
                echo "✅ Installed... python3.12-venv"
        else
                echo "⚠️ Installing python3.12-venv"
                sudo apt install python3.12-venv
        fi

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

#### Build the Env Box	
	# 1. Create a virtual environment
		python3 -m venv ./.venv

	# 2. Activate it
		source ./.venv/bin/activate

	# 3. Update
		pip install --upgrade pip
fi



if [ "$PIP_INSTALL" == True ];then
        #if echo "$VIRTUAL_ENV_PROMPT"|grep -q '.venv' ];then
        #        echo "✅ Sourced"
        #else
                source ./.venv/bin/activate
        #fi


        if echo "$APT_LIST"|grep -q portaudio19-dev;then
                echo "✅ Installed... portaudio19-dev"
        else
                echo "⚠️ Install portaudio19-dev for audio"
                sudo apt install portaudio19-dev
        fi

        if echo "$APT_LIST"|grep -q ffmpeg;then
                echo "✅ Installed... ffmpeg"
        else
                echo "⚠️ Install ffmpeg for audio playback"
                sudo apt install ffmpeg
        fi
        
#### Audio/Voice
	#pip install pyaudio  ## Legacy
        #pip install pyttsx3  ## builtin voice
        pip install sounddevice scipy faster-whisper
        pip install gTTS
	
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

touch .venv/$RUN
fi




#### Run the Box
#if echo "$VIRTUAL_ENV_PROMPT"|grep -q '.venv' ];then
#                echo "✅ Sourced"
#        else
                source ./.venv/bin/activate
#fi	
#### Export Variables
        export PYTHONWARNINGS="ignore"
#### Run the AI
	echo "Starting the AI"
	python -m webui
	exit
