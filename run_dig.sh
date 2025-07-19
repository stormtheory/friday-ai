#!/bin/bash
cd "$(dirname "$0")"

# Written by StormTheory
# https://github.com/stormtheory/friday-ai

### Creates or opens the virtual Enviorment needed for AI tools to run
##### Note you will need at least 4G of /tmp space available for the startup install.
##### Virtual environment may take up 7Gbs of space for all needed packages.
##### Runs the creating and installing of the virtual environment setup one time.

PYENV_DIR='./.venv'
RUN='.run_dig_installed'

# No running as root!
ID=$(id -u)
if [ "$ID" == '0'  ];then
        echo "Not safe to run as root... exiting..."
        exit
fi

# 🛡️ Set safe defaults
set -euo pipefail
IFS=$'\n\t'

# 🧾 Help text

show_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -w, --webui       WebUI GUI
  -l, --local       Local TK GUI (default)
  --i2i             Run Image to Image Generator (I2I)
  -d, --debug       Debug mode
  -h, --help        Show this help message
EOF
}

# Default values
WEBUI=false
LOCAL_TK=true
DEBUG=false
RUN_I2I=false
RUN_POSTFIX='-local'

# Parse options with getopt
OPTIONS=$(getopt -o wldh --long webui,local,i2i,debug,help -- "$@")
if [ $? -ne 0 ]; then
  show_help
  exit 1
fi

eval set -- "$OPTIONS"

while true; do
  case "$1" in
    -w|--webui)
      WEBUI=true
      LOCAL_TK=false
      RUN_POSTFIX='-webui'
      shift ;;
    -l|--local)
      LOCAL_TK=true
      WEBUI=false
      RUN_POSTFIX='-local'
      shift ;;
    --i2i)
      RUN_I2I=true
      LOCAL_TK=false
      WEBUI=false
      RUN_POSTFIX='-i2i'
      shift ;;
    -d|--debug)
      DEBUG=true
      shift ;;
    -h|--help)
      show_help
      exit 0 ;;
    --)
      shift
      break ;;
    *)
      echo "❌ Invalid option: $1"
      show_help
      exit 1 ;;
  esac
done


if [ ! -d $PYENV_DIR ];then
        APT_LIST=$(apt list 2>/dev/null)
        ENV_INSTALL=True
        PIP_INSTALL=True
elif [ -f $PYENV_DIR/$RUN ];then
        echo "✅ Installed... .venv"
        echo "✅ Installed... $RUN"
        ENV_INSTALL=False
        PIP_INSTALL=False
elif [ ! -f $PYENV_DIR/$RUN ];then
	echo "✅ Installed... .venv"
        APT_LIST=$(apt list 2>/dev/null)
        ENV_INSTALL=False
        PIP_INSTALL=True
else
        exit
fi

if [ "$ENV_INSTALL" == 'True' ];then
### Checking dependencies

	APT_LIST=$(apt list 2>/dev/null)
        if echo "$APT_LIST"|grep python3.12-dev;then
                echo "✅ Installed... python3.12-dev"
        else
                echo "⚠️ Installing python3.12-dev"
                sudo apt install python3.12-dev
        fi

	if echo "$APT_LIST"|grep python3.12-venv;then
		echo "✅ Installed... python3.12-venv"
	else
		echo "⚠️ Installing python3.12-venv"
		sudo apt install python3.12-venv
	fi

	if echo "$APT_LIST"|grep nvidia-driver;then
		echo "✅ Installed... nvidia-driver"
		if echo "$APT_LIST"|grep nvidia-cuda-toolkit;then
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
		python3 -m venv $PYENV_DIR

	# 2. Activate it
		source $PYENV_DIR/bin/activate

	# 3. Update
		pip install --upgrade pip
fi



if [ "$PIP_INSTALL" == True ];then
    source $PYENV_DIR/bin/activate

#### Image Generaters
	# For CUDA 11.8 (check your version: nvidia-smi)
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	
	# General Image Gen
	pip install --upgrade pip setuptools wheel
	pip install diffusers[torch] transformers[vision] accelerate safetensors
	pip install torch transformers accelerate
	pip install diffusers transformers accelerate safetensors
	pip install xformers

	# ⚠️ bitsandbytes works best with NVIDIA GPUs. For CPU-only, consider ggml or ctransformers.
	## CPU RAM Offload / XL Image Gen
	pip install bitsandbytes
	pip install git+https://github.com/huggingface/huggingface_hub.git

touch $PYENV_DIR/$RUN
fi

#### Run the Box
	source $PYENV_DIR/bin/activate


if [ $WEBUI == true ]; then
	#### Check dependancies
            if [ ! -f "$PYENV_DIR/$RUN$RUN_POSTFIX" ];then
                    pip install gradio  # WebGUI
                    touch $PYENV_DIR/$RUN$RUN_POSTFIX
            fi
	#### Export Variables
		export PYTHONWARNINGS="ignore"
		export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	#### Run the AI
		echo "Starting the Detailed Image Gen (DIG) WebUI"
		python -m DIG-webUI
		exit 0
elif [ $LOCAL_TK == true ];then
	#### Check dependancies
                if [ ! -f "$PYENV_DIR/$RUN$RUN_POSTFIX" ];then
                    APT_LIST=$(apt list 2>/dev/null)
                    if echo "$APT_LIST"|grep python3-tk;then
                            echo "✅ Installed... python3-tk"
                    else
                            echo "⚠️ Installing python3-tk"
                            sudo apt install python3-tk
                    fi
                    pip install customtkinter
                    pip install pillow # Icon
                    touch $PYENV_DIR/$RUN$RUN_POSTFIX
                fi
	#### Export Variables
		export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	#### Run the AI
		echo "Starting the Detailed Image Gen (DIG) Tinkter"
		python -m DIG-tk
		exit 0
elif [ $RUN_I2I == true ];then
	#### Check dependancies
                if [ ! -f "$PYENV_DIR/$RUN$RUN_POSTFIX" ];then
                    APT_LIST=$(apt list 2>/dev/null)
                    if echo "$APT_LIST"|grep python3-tk;then
                            echo "✅ Installed... python3-tk"
                    else
                            echo "⚠️ Installing python3-tk"
                            sudo apt install python3-tk
                    fi
                    pip install customtkinter
                    pip install pillow # Icon
                    touch $PYENV_DIR/$RUN$RUN_POSTFIX
                fi
	#### Export Variables
		export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	#### Run the AI
		echo "Starting the Image to Image Gen (I2I) Tinkter"
		python -m img2img-transformer
		exit 0
fi
echo "ERROR!"
exit 1