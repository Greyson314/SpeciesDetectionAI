# Senior Comprehensive Project
## Author: Grey Hutchinson

<p align="center">
<img src="https://www.allaboutbirds.org/guide/assets/photo/299890491-1280px.jpg"
     alt="Common Tern"
     style="width: 200px; margin-left: auto; margin-right: auto" />
</p>
My project, which I nicknamed “Murmur”, was to create a research tool that would use neural networks, which can read and learn patterns to try and classify an audio file, to analyze any given PAM file and return a list of timestamps where the target species made a vocalization.

### How to: Install this project

1. Make sure you have python3.6 or greater installed on you computer
2. Create a [virtual environment](https://docs.python.org/3/library/venv.html)
   1. `python -m venv venv`
   2. Windows: `./venv/Source/activate` 
3. Install [requirements.txt](requirements.txt)
   1. `pip install -r requirements.txt`
4. Separate your 

### How to: Run this project with passive acoustic monitoring files

1.  Extract saved_model.zip to this directory, making sure the folder is called "saved_model".
2.  Save the PAM file to the root directory (e.g. `pam_1.wav`)
3.  Go into [app.py](app.py) and change line 129 to match your file name
4.  Run app.py

### How to: Use this to create your own model

1. You will need to use [murmur.py](src/murmur.py) to create the model.
