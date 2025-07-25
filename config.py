# Written by StormTheory
# https://github.com/stormtheory/friday-ai

from pathlib import Path

ENABLE_SPEECH_INPUT = False
ENABLE_SPEECH_OUTPUT = False

### Titles and Banners
WEBUI_TITLE = 'Friday'                                    # Not currently working
WEBUI_TOP_PAGE_BANNER = '🤖 Friday — Your Local AI Assistant'
WEBUI_CHATBOT_LABEL = 'Friday'
WEBUI_SPEAK_TO_TEXT_LABEL = '🎤 Speak to Friday'

CHATBOT_TITLE = 'Friday — Your Local AI Assistant'
CHATBOT_LABEL = 'Friday'

DIG_WEBUI_TITLE = 'Friday Image Gen'                       # Not currently working 
DIG_WEBUI_TOP_PAGE_BANNER = ' 🎨 Stable Diffusion Image Generator'
DIG_WEBUI_FILENAME = 'detailed_XL_friday'  # Will look something like: detailed_XL_friday_{timestamp}.png

CLI_FRIDAY_WELCOME_MESSAGE = '👋 FRIDAY Initialized: Friendly Responsive Intelligent Digital Assistant for You'
CLI_FRIDAY_EXIT_MESSAGE = 'Goodbye! Have a productive day.'

### LLM Prompts
DIG_DEFAULT_PROMPT = "a samurai standing on Mars with a red sun"
DEFAULT_LLM_MODEL = 'mistral-ollama'         # [ mistral-ollama or llama3 or mistral-raw ]
USER_PROMPT_NAME = 'You'
ASSISTANT_PROMPT_NAME = 'Friday'

### Pre-Prompts
LLAMA3_PRE_PROMPT = 'You are Friday, a helpful, concise, AI assistant.' 
MISTRAL_PRE_PROMPT = 'You are Friday, a helpful, concise, AI assistant. If the user rejects a suggestion, do NOT repeat the same idea. Instead, politely acknowledge and ask how else you can assist.'

DIG_PICTURE_NEG_PROMPT = 'blurry, low quality, distorted, artifacts, text, watermark, extra limbs, deformed hands, extra fingers, broken anatomy' # way to tell the AI what you don’t want to see
DIG_PICTURE_HEIGHT = 1024              # Hieght of the images to be generated in pixels - The bigger the more resources needed
DIG_PICTURE_WIDTH = 1024               # Width of the images to be generated in pixels - The bigger the more resources needed
DIG_PICTURE_GUIDANCE_SCALE=7.5         # [1.0 to 20.0] Controls how strongly the model should follow your text prompt.
DIG_PICTURE_NUM_INFERENCE_STEPS = 25   # [20 to 100] Number of denoising steps the model takes to generate the image. 
                                       #     25–50 is often enough for good results. 
                                       #     60+ only if you need high-res or cleaner details

### Other DIG Settings
DIG_DEFAULT_GEN_LOOP_TIMES = 1

    ####################
###### SAVE LOCATIONS ######
    ####################

### Image Save locations
DIG_WEBUI_IMAGE_SAVE_HOMESPACE_LOCATION = 'Pictures/AI'
IMAGE_GEN_IMAGE_SAVE_HOMESPACE_LOCATION = 'Pictures/AI'
DIG_WEBUI_THREAD_DATA_DIR = str(Path.home() / ".friday_ai_data" / "detail_image_gen")

### CLI Input History
CLI_HISTORY_FILE = str(Path.home() / ".friday_ai_data" / "cli_history")

### Global Memory
MEMORY_FILE = str(Path.home() / ".friday_ai_data" / "memory.json")

### Threads - This is for chatbox display
THREADS_DIR = str(Path.home() / ".friday_ai_data" / "chatbox_display_threads")
ACTIVE_FILE = str(Path.home() / ".friday_ai_data" / "active_thread.json")

### Context
CONTEXT_DIR = str(Path.home() / ".friday_ai_data" / "context")
MAX_HISTORY = 10



    ####################
######     COLORS     ######
    ####################

# ANSI colors
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[94m"
GREEN = "\033[92m"
RESET = "\033[0m"

YELLOW_BG = "\033[43m"
RED_BG = "\033[41m"
