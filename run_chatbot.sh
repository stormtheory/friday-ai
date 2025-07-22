#!/bin/bash
cd "$(dirname "$0")"

# Written by StormTheory
# https://github.com/stormtheory/friday-ai

### Creates or opens the virtual Enviorment needed for AI tools to run
##### Note you will need at least 4G of /tmp space available for the startup install.
##### Virtual environment may take up 7Gbs of space for all needed packages.
##### Runs the creating and installing of the virtual environment setup one time.

PYENV_DIR='./.venv'
RUN='.run_chatbot_installed'

# No running as root!
ID=$(id -u)
if [ "$ID" == '0'  ];then
        echo "Not safe to run as root... exiting..."
        exit
fi

# CAN ONLY BE ONE!!!!
APP='chatbot'
RAM_DIR='/dev/shm'
BASENAME=$(basename $0)
RAM_FILE="${RAM_DIR}/${APP}-${BASENAME}.lock"
fs_type=$(stat -f -c %T "$RAM_DIR")
if [ -d $RAM_DIR ];then
        if [[ "$fs_type" == "tmpfs" ]] || [[ "$fs_type" == "ramfs" ]]; then
                if [ -f $RAM_FILE ]; then
                echo "RAM lock file exists: $RAM_FILE"
                exit
                else
                        touch $RAM_FILE
                        chmod 600 $RAM_FILE
                        # Cleanup on exit
                        trap 'rm -f "$RAM_FILE"; echo "[*] Lock released."; exit' INT TERM EXIT
                fi
        else
                echo "[-] '$RAM_DIR' is NOT on a RAM disk (type: $fs_type)"
        fi
else
        echo "ERROR: $RAM_DIR not present to lock app."
fi


# 🛡️ Set safe defaults
set -euo pipefail
IFS=$'\n\t'

# 🧾 Help text
show_help() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -w             WebUI GUI
  -l             Local TK GUI (Default)
  -c             Command Line Interface (CLI)
  -d             Debug mode
  -h             Show this help message

Example:
  $0 -vdl
EOF
}

# 🔧 Default values
WEBUI=false
LOCAL_TK=true
CHAT_CLI=false
DEBUG=false
RUN_POSTFIX='-local'

# 🔍 Parse options
while getopts ":wldhc" opt; do
  case ${opt} in
    c)
        WEBUI=true
        LOCAL_TK=false
        CHAT_CLI=true
        RUN_POSTFIX='-cli'
        ;;
    w)
        WEBUI=true
        LOCAL_TK=false
        CHAT_CLI=false
        RUN_POSTFIX='-webui'
        ;;
    l)
        LOCAL_TK=true
	WEBUI=false
        CHAT_CLI=false
        RUN_POSTFIX='-local'
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
        exit 1
fi

if [ "$ENV_INSTALL" == 'True' ];then
### Checking dependencies
        
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

        if echo "$APT_LIST"|grep portaudio19-dev;then
                echo "✅ Installed... portaudio19-dev"
        else
                echo "⚠️ Install portaudio19-dev for audio"
                sudo apt install portaudio19-dev
        fi

        if echo "$APT_LIST"|grep festival;then
                echo "✅ Installed... festival"
        else
        read -p "⚠️ Install festival for private voice playback? [y] > " ANS
                if [ "$ANS" == y ];then
                        sudo install festival
                fi
        fi

        if echo "$APT_LIST"|grep ffmpeg;then
                echo "✅ Installed... ffmpeg"
        else
                echo "⚠️ Install ffmpeg for audio playback"
                sudo apt install ffmpeg
        fi
        
#### Audio/Voice
	#pip install SpeechRecognition ## Legacy
        #pip install pyaudio  ## Legacy
        #pip install pyttsx3  ## builtin voice ## Robotic
        #pip install gTTS # Not private from Google
        #pip install edge-tts # Not private from Microsoft
        pip install sounddevice scipy faster-whisper ## For SpeechRecognation

	
#### Image Generaters
	# For CUDA 11.8 (check your version: nvidia-smi)
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	pip install --upgrade pip setuptools wheel
	pip install diffusers[torch] transformers[vision] accelerate safetensors
	pip install torch transformers accelerate
	pip install diffusers transformers accelerate safetensors
	pip install xformers

	pip install pymupdf # PDF

#### Indexing / RAG
	pip install sentence_transformers
	pip install langchain
	pip install sentence_transformers
	pip install faiss-cpu   # Uses the CPU for RAG
	#pip install faiss-gpu  # Uses the GPU for RAG - Have ran into alot of issues

touch $PYENV_DIR/$RUN
fi



#### Run the Box
        source $PYENV_DIR/bin/activate

if [ $CHAT_CLI == true ]; then
        #### Export Variables
                export PYTHONWARNINGS="ignore"
        #### Run the AI
                echo "Starting the AI"
                python3 -m cli-main
                exit 0
elif [ $WEBUI == true ]; then
        #### Check dependancies
                if [ ! -f "$PYENV_DIR/$RUN$RUN_POSTFIX" ];then
                        pip install gradio  # WebGUI
                        touch $PYENV_DIR/$RUN$RUN_POSTFIX
                fi
        #### Export Variables
                export PYTHONWARNINGS="ignore"
        #### Run the AI
                echo "Starting the AI"
                python3 -m chatbot-webui
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
		export PYTHONWARNINGS="ignore"
	#### Run the AI
		echo "Starting the AI"
		python3 -m chatbot-local
		exit 0
fi
echo "ERROR!"
exit 1
