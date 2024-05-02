# Code for bird call detection model based on Detr

![image info](docs/assets/demo_results/emb_ort.png)

### Prerequisit
- Install python version 3.10

### Install
Create virtual environment named .venv with conda (or venv) and activate it.
Here is how to do with conda.

```
conda create .venv
conda activate .venv
pip install -r requirements.txt
```

and with venv

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**Do not forget to install ffmpeg and to add it to your path as described [here](https://phoenixnap.com/kb/ffmpeg-windows)**

### Download model weights
You need to manually download [here](https://drive.google.com/drive/folders/1gMoLpgnpGw2c15mVVyN6W6e4FDHCD24n?usp=sharing) the model weights and the pre-trained model contained in the .zip called `models.zip` and put the content in a folder called `model` at the root of the project.

### Download example image to launch the inference notebook
Donwload the .zip called `data.zip` [here](https://drive.google.com/drive/folders/1gMoLpgnpGw2c15mVVyN6W6e4FDHCD24n?usp=sharing) and put its content in a folder called `data` at the root of the project.

### Run inference notebook
Open `notebooks/1.0_LB_exemple-inference.ipynb` and run each cell one after the other.



