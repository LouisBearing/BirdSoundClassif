from fastapi import FastAPI, File, UploadFile
import os

app = FastAPI()

# Chemin vers le dossier de stockage des enregistrements
RECORDS_FOLDER = "records"

# Créez le dossier s'il n'existe pas déjà
os.makedirs(RECORDS_FOLDER, exist_ok=True)

@app.post("/upload/")
async def upload_record(file: UploadFile = File(...)):
    # Vérifiez si le fichier est un fichier audio .wav ou .mp3
    if file.content_type not in ["audio/wav", "audio/mp3"]:
        return {"error": "Le fichier doit être un fichier audio .wav ou .mp3"}

    # Créez le chemin complet pour stocker le fichier dans le dossier records
    file_path = os.path.join(RECORDS_FOLDER, file.filename)

    # Écrivez le contenu du fichier dans le dossier records
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return {"filename": file.filename, "message": "Fichier enregistré avec succès"}