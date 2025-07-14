# friday-ai
FRIDAY (Friendly Responsive Intelligent Digital Assistant for You)
F.R.I.D.A.Y. is a lightweight, self-hosted AI environment I’m building under the name alias Stormtheory—designed to offer the power of GPT-like models with full privacy, locally ran, with security, and user control at its core.

Created for ease of use and respect for your data, F.R.I.D.A.Y. runs entirely from home—no cloud, no tracking, no compromise—just a fast, private AI that works for you, not on you.

# In the Works
Local API to GPT4 to allow for more tokens and more work to get done faster. UI enhancements and adding stable-diffusion-xl-base-1.0 as an option from the main webUI. More stored documents and better referancing using the local API calls.

# System Requirements
At this time Ubuntu/Mint is only tested to be supported, but could work on RHEL/Rocky/CentOS, no Yum/DNF package support yet. Please feedback if you want a YUM/DNF .rpm package. If there is interest in other Linux flavors/families please let me know or it's just a project for me and my family :P as our daily drivers. 

# INSTALL
1) Download the latest released .deb package file off of github and install on your system.
2) Build DEB Install file:
	
	Download the zip file of the code, off of Github. This is found under the [<> Code] button on https://github.com/stormtheory/friday-ai.
	
	Extract directory from the zip file. Run the build script in the directory. 

        ./build

   	Install the outputted .deb file.

3) Install without Package Manager, run commands:

	Download the zip file of the code, off of Github. This is found under the [<> Code] button on https://github.com/stormtheory/friday-ai.

	Extract directory from the zip file. Run the following commands within the directory.

        # Install script for llama3 LLM
        friday/install_ollama.sh

        # Run any of the following
        friday/run_cli_friday.sh
        friday/run_detailed_webui_image_gen.sh
        friday/run_webui_friday.sh


# Overview of Data Flow


# User Agreement:
This project is not a company or business. By using this project’s works, scripts, or code know that you, out of respect are entitled to privacy to highest grade. This product will not try to steal, share, collect, or sell your information. However 3rd parties such at Github may try to use your data without your consent. Users or admins should make reports of issue(s) related to the project’s product to the project to better equip or fix issues for others who may run into the same issue(s). By using this project’s works, scripts, code, or ideas you as the end user or admin agree to the GPL-2.0 License statements and acknowledge the lack of Warranty. As always, give us a Star on Github if you find this useful, and come help us make it better.

As stated in the GPL-2.0 License:
    "This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details." Also "ABSOLUTELY NO WARRANTY".
