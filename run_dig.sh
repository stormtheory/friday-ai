#!/bin/bash
cd "$(dirname "$0")"

# Written by StormTheory
# https://github.com/stormtheory/friday-ai

### Creates or opens the virtual Enviorment needed for AI tools to run
##### Note you will need at least 4G of /tmp space available for the startup install.
##### Virtual environment may take up 7Gbs of space for all needed packages.
##### Runs the creating and installing of the virtual environment setup one time.

RUN='.run_detail_image_gen_installed'

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
  -w             WebUI GUI    (Default)
  -l             Local TK GUI
  -d             Debug mode
  -h             Show this help message

Example:
  $0 -vdl
EOF
}

# 🔧 Default values
WEBUI=false
LOCAL_TK=true
DEBUG=false

# 🔍 Parse options
while getopts ":wldh" opt; do
  case ${opt} in
    w)
      WEBUI=true
	  LOCAL_TK=false
      ;;
    l)
      LOCAL_TK=true
	  WEBUI=false
      ;;
    d)
      DEBUG=true
      ;;
    h)
      show_help
      exit 0
      ;;
    \?)
      echo "❌ Invalid option: -$OPTARG" >&2
      show_help
      exit 1
      ;;
    :)
      echo "❌ Option -$OPTARG requires an argument." >&2
      show_help
      exit 1
      ;;
  esac
done



if [ ! -d ./.venv ];then
        APT_LIST=$(apt list 2>/dev/null)
        ENV_INSTALL=True
        PIP_INSTALL=True
elif [ -f ./.venv/$RUN ];then
        echo "✅ Installed... .venv"
        echo "✅ Installed... $RUN"
        ENV_INSTALL=False
        PIP_INSTALL=False
elif [ ! -f ./.venv/$RUN ];then
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
		python3 -m venv ./.venv

	# 2. Activate it
		source ./.venv/bin/activate

	# 3. Update
		pip install --upgrade pip
fi



if [ "$PIP_INSTALL" == True ];then
    source ./.venv/bin/activate

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

touch .venv/$RUN
fi

#### Run the Box
	source ./.venv/bin/activate


if [ $WEBUI == true ]; then
		pip install gradio # WebUI
	#### Export Variables
		export PYTHONWARNINGS="ignore"
		export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	#### Run the AI
		echo "Starting the Detailed Image Gen (DIG) WebUI"
		python -m DIG-webUI
		exit 0
elif [ $LOCAL_TK == true ];then
	#### Check dependancies
		APT_LIST=$(apt list 2>/dev/null)
		if echo "$APT_LIST"|grep python3-tk;then
			echo "✅ Installed... python3-tk"
		else
			echo "⚠️ Installing python3-tk"
			sudo apt install python3-tk
		fi
	#### Export Variables
		export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
	#### Run the AI
		echo "Starting the Detailed Image Gen (DIG) Tinkter"
		python -m DIG-tk
		exit 0
fi
echo "ERROR!"
exit 1




