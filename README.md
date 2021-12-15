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

### How to: Run this project with passive accustic audio files

<!-- TODO: where ya gunna host that model?? -->
1.  Download the model from [here](.) 
2.  Extract it to this directory, making sure the folder is called, "saved_model".
3.  Save the bird file to the root directory (like `pam_1.wav`)
4.  Go into [app.py](app.py) and change line 129 to match your file name
5.  Run app.py

### How to: Use this to create your own model

1. You will need to use [murmer.py](src/murmur.py) to create the model.
2. Within the head of murmer.py, there are instructions on how to use that file to generate the needed resources
