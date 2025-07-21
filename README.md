# friday-ai:
FRIDAY (Friendly Responsive Intelligent Digital Assistant for You)
F.R.I.D.A.Y. is a lightweight, self-hosted AI environment I’m building under the name alias Stormtheory — designed to offer the power of GPT-like models with full privacy, locally hosted, with security, and user control at its core.

Created for ease of use and respect for your data, F.R.I.D.A.Y. runs entirely from home — no cloud, no tracking, no compromise—just a fast, private AI that works for you, not on you.

# Image Screenshots:
## Flagship Products
### Chatbot Local Window App
<img width="960" height="748" alt="Image" src="https://github.com/user-attachments/assets/820f13e3-5f78-4cc2-a1ba-72de8642f711" />

### Detailed Image Generator (DIG) Local
<img width="700" height="898" alt="Image" src="https://github.com/user-attachments/assets/860d3dae-d697-49f5-8742-ff35baaafc84" />

### Image to Image Generator (I2I) Local
<img width="900" height="828" alt="Image" src="https://github.com/user-attachments/assets/b241580d-0e9e-4f55-a191-eca635b375e1" />


# In the Works:
Guiding Principle(s):
- As always: Fighting to keep this AI free, private, opensource, fast, and easy (in that order).

### Ultimate Goals
- A free, private, fast, and easy in home AI that can do tasks outside of just being a chatbot.
- Change the temperature in the home, set calender events, hold a complex conversation, set reminders, along with many other features and tasks we see in AI apps now.
- Integrate into a smart speaker but 100% private and free.
- A home/small business(watch your licensing) network monitor and defender.

### Roadmap
[✔] Started: July 2025
[✔] ChatBot
[✔] Image Generator (DIG)
[✔] Image to Image Generator (I2I)
[ ] Storyteller
[ ] Taking a model to use it to run commands on your computer in a sandboxed environment.
[ ] Home automation
[ ] Building Friday's own LLM model
[ ] Training Friday's own LLM model
[ ] Fine tuning Friday's own LLM model

### Goals
Short-term:
- Making a chat bot that can tell a story while making illustrations with the image generator.
- Add debug and reduce output.

Long-term:
- Getting a Female Voice (Running into privacy issues) Did this with gTTS but AI responses were being routed to Google and others to Microsoft.

Upon Request:
- Adding audio record Local Tkinter Chatbot.
- Add YUM support.
- Add AMD support.

# System Requirements:
- Ubuntu/Mint is only tested to be supported.
- Nvidia GPU(s) and nvidia-driver but AMD can be added.
	- Tested with a 8GB NVIDIA RTX 3060 TI
- At least 10GBs or 14GBs of homespace depending on the number of AI models
- At least 30GBs of Harddrive space
- At least 5GBs of /tmp space (first time install)

Friday could work on RHEL/Rocky/CentOS, no Yum/DNF package support yet. 
Please feedback if you want a YUM/DNF .rpm package. 
If there is interest in other Linux flavors/families please let me know or it's just a project for me and my family :P as our daily drivers. 

# INSTALL:
 Run scripts will create(if not present) or open the virtual Enviorment needed for AI tools to run.
 Note you will need at least 4G of /tmp space available for the first time startup install.
 Virtual environment may take up 7Gbs of space for all needed packages.

1) Download the latest released .deb package file off of github and install on your system.
2) Build DEB Install file:
	
	Download the zip file of the code, off of Github. This is found under the [<> Code] button on https://github.com/stormtheory/friday-ai.
	
	Extract directory from the zip file. Run the build script in the directory. 

        ./build

   	Install the outputted .deb file.

3) Install without Package Manager, run commands:

	Download the zip file of the code, off of Github. This is found under the [<> Code] button on https://github.com/stormtheory/friday-ai.

	Extract directory from the zip file. Run the following commands within the directory.

        # Install script for llama3 LLM:
        friday/install_ollama.sh

        # If you want to try the French Fully-Opensource LLM Mistral then:
        friday/install_mistral_from_TheBloke.sh

# RUN:
### run the local Windowed App

        # Detailed Image Generator (DIG)
        ./run_dig.sh -l

        # Chatbot
        ./run_chatbot.sh -l

### run the CLI

        # Command Line Interface (CLI)
        ./run_chatbot.sh -c

### run the WebUI

        # Detailed Image Generator (DIG)
        ./run_dig.sh -w
        firefox http://127.0.0.1:7860

        # Chatbot
        ./run_chatbot.sh -w
        firefox http://127.0.0.1:7860

# Tips/Tricks/File Locations:
All images generated are saved to your Pictures directory in your homespace un a folder called AI:
        
	cd ~/Pictures/AI/ ; ls

All saved data other then pictures are saved in ~/.friday_ai_data/ unless changed in config.py from the default.

        cd ~/.friday_ai_data/   ### Saved your settings, data, and history.

Pip install location:

        ./.venv     # within the friday-ai dir

Huggingface models and data is saved:

        ~/.cache/huggingface

Ollama files are found here:

        /etc/systemd/system/ollama.service
        /usr/bin/ollama
        /usr/lib/ollama/
        /usr/share/ollama/
        /usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/  # Saved LLM location

By default the webUI after ran can be found by running this command:
 	
  	firefox http://127.0.0.1:7860

# More Image Screenshots:
### Command Line Interface (CLI)
<img width="958" height="224" alt="Image" src="https://github.com/user-attachments/assets/cda0f84b-b81d-48dd-a337-e8c6ec7dec40" />


## Local Web Browser Based
### Detailed Image Generator Local webUI
<img width="774" height="791" alt="Image" src="https://github.com/user-attachments/assets/002a07aa-40cf-44e8-8d79-16766abfc461" />

#### Just the 'Advanced Settings'
<img width="987" height="366" alt="Image" src="https://github.com/user-attachments/assets/b2d88abb-d707-4eac-bf73-83c8560a772b" />


## friday_local_webUI
<img width="1494" height="886" alt="Image" src="https://github.com/user-attachments/assets/54f8d17f-bc45-44ff-9318-8ace9d07f1fc" />

# User Agreement:
This project is not a company or business. By using this project’s works, scripts, or code know that you, out of respect are entitled to privacy to highest grade. This product will not try to steal, share, collect, or sell your information. However 3rd parties such at Github may try to use your data without your consent. Users or admins should make reports of issue(s) related to the project’s product to the project to better equip or fix issues for others who may run into the same issue(s). By using this project’s works, scripts, code, or ideas you as the end user or admin agree to the GPL-2.0 License statements and acknowledge the lack of Warranty. As always, give us a Star on Github if you find this useful, and come help us make it better.

As stated in the GPL-2.0 License:
    "This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details." Also "ABSOLUTELY NO WARRANTY".
