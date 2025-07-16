# friday-ai:
FRIDAY (Friendly Responsive Intelligent Digital Assistant for You)
F.R.I.D.A.Y. is a lightweight, self-hosted AI environment I’m building under the name alias Stormtheory — designed to offer the power of GPT-like models with full privacy, locally ran, with security, and user control at its core.

Created for ease of use and respect for your data, F.R.I.D.A.Y. runs entirely from home — no cloud, no tracking, no compromise—just a fast, private AI that works for you, not on you.

# In the Works:
Local API to GPT4 to allow for more tokens and more work to get done faster. UI enhancements and adding stable-diffusion-xl-base-1.0 as an option from the main webUI. More stored documents and better referancing using the local API calls.

# System Requirements:
- Ubuntu/Mint is only tested to be supported.
- Nvidia GPU(s) and nvidia-driver but AMD can be added.
	- Tested with a 8GB NVIDIA RTX 3060 TI
- At least 8GBs or 14GBs of homespace depending on the number of AI models
- At least 20GBs of Harddrive space
- At least 4GBs of /tmp space (first time install)

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

        # Run any of the following depending on what you want to do.
        ## Starts the webUI for the Image Generator
        friday/run_detailed_webui_image_gen.sh
        ## Starts the Command Line Interface (CLI) Version of Friday
        friday/run_cli_friday.sh
        ## Starts the webUI Version of Friday
        friday/run_webui_friday.sh

# Tips/Tricks:
All images generated are saved to your Pictures directory in your homespace un a folder called AI:
        
	cd ~/Pictures/AI/ ; ls
 
 By default the webUI after ran can be found by running this command:
 	
  	firefox http://127.0.0.1:7860

# Image Screenshots:
### Detailed Image Generator Local webUI
<img width="774" height="791" alt="Image" src="https://github.com/user-attachments/assets/002a07aa-40cf-44e8-8d79-16766abfc461" />

#### Just the 'Advanced Settings'
<img width="987" height="366" alt="Image" src="https://github.com/user-attachments/assets/b2d88abb-d707-4eac-bf73-83c8560a772b" />

## friday_cli
<img width="654" height="194" alt="Image" src="https://github.com/user-attachments/assets/87a2fb25-27e0-4049-94a6-e4ab9b85302b" />

## friday_local_webUI
<img width="1280" height="906" alt="Image" src="https://github.com/user-attachments/assets/aa1c13e8-5ae0-43c5-aa62-842765d42392" />

# User Agreement:
This project is not a company or business. By using this project’s works, scripts, or code know that you, out of respect are entitled to privacy to highest grade. This product will not try to steal, share, collect, or sell your information. However 3rd parties such at Github may try to use your data without your consent. Users or admins should make reports of issue(s) related to the project’s product to the project to better equip or fix issues for others who may run into the same issue(s). By using this project’s works, scripts, code, or ideas you as the end user or admin agree to the GPL-2.0 License statements and acknowledge the lack of Warranty. As always, give us a Star on Github if you find this useful, and come help us make it better.

As stated in the GPL-2.0 License:
    "This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details." Also "ABSOLUTELY NO WARRANTY".
