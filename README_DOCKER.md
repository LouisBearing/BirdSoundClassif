

### Remarques

#### Structure des dossiers
- Les différents services ont chacun leur sous-dossier dans `app/`:
  - `inference/`
  -  `api/`
  -  ...
- Les sous dossiers de `docker/` reprennent cette même structure


#### Le dossier docker
- L'image **base** sert de base d'image pour construire les autres, elle comprend python et notre package custom **src**
- Les autres images ont chacune leur propre jeu de dépendances (`requirements.txt`) pour éviter des images inutilement lourdes:
  - **Api**: fastapi, requests etc
  - **Inference**: torch, ffmpeg, librosa etc
- Dans chaque dossier d'image, un script `build.sh` permet de construire l'image en allant chercher le contenu correspondant dans `app/`
- Le tag des images reprend mes identifiants Dockerhub: `username/repo_name:local_folder_name`, remplacer ce tag par vos propres identifiants dans les fichiers `build.sh` si nécessaire
- Si vous voulez pull un image sur dockerhub, les scripts `pull.sh` doivent être modifiés du coup


### Installation

```bash
cd docker/base
./build.sh

cd docker/inference
./build.sh

```

### Lancement du container
Lancement du container en mode bash

```bash
# Si necessaire:
cd docker/inference

# Puis:
./start_bash.sh

```

Dans le container:
```bash
python inference/main.py


#===================
# Commandes de debug
# Check des libraries python:
pip freeze | cat


```



### Après un premier essai...
On retourne au bon vieux message d'erreur dû aux problèmes de cpu... 
Il semble que l'on doit repackager un 2ème `src` avec certaines parties du code réécrites pour le cpu

```bash
root@2ca153072c27:/app# python inference/main.py 
Traceback (most recent call last):
  File "/app/inference/main.py", line 10, in <module>
    model, config = load_model(mod_p)
  File "/app/src/models/run_detection.py", line 84, in load_model
    model = load_weights(config, model, path=os.path.join(mod_p, 'model_chkpt_last.pt'), train=False).to(config.device)
  File "/app/src/models/detr.py", line 355, in load_weights
    state_dict = torch.load(path)
  File "/usr/local/lib/python3.10/dist-packages/torch/serialization.py", line 1026, in load
    return _load(opened_zipfile,
  File "/usr/local/lib/python3.10/dist-packages/torch/serialization.py", line 1438, in _load
    result = unpickler.load()
  File "/usr/local/lib/python3.10/dist-packages/torch/serialization.py", line 1408, in persistent_load
    typed_storage = load_tensor(dtype, nbytes, key, _maybe_decode_ascii(location))
  File "/usr/local/lib/python3.10/dist-packages/torch/serialization.py", line 1382, in load_tensor
    wrap_storage=restore_location(storage, location),
  File "/usr/local/lib/python3.10/dist-packages/torch/serialization.py", line 391, in default_restore_location
    result = fn(storage, location)
  File "/usr/local/lib/python3.10/dist-packages/torch/serialization.py", line 266, in _cuda_deserialize
    device = validate_cuda_device(location)
  File "/usr/local/lib/python3.10/dist-packages/torch/serialization.py", line 250, in validate_cuda_device
    raise RuntimeError('Attempting to deserialize object on a CUDA '
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
```


