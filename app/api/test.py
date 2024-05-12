import unittest
import requests

class TestUploadAPI(unittest.TestCase):
    def test_valid_file(self):
        # URL de l'API
        url = "http://127.0.0.1:8000/upload/"
        
        # Chemin du fichier .wav à envoyer
        file_path = "Test\Turdus_merlula.wav"
        
        # Ouvrir le fichier en mode lecture binaire
        with open(file_path, "rb") as file:
            files = {"file": ("audio_file.wav", file, "audio/wav")}
            response = requests.post(url, files=files)
        
        # Vérifier la réponse
        self.assertEqual(response.status_code, 200)
        self.assertIn("filename", response.json())
        self.assertEqual(response.json()["message"], "Fichier enregistré avec succès")
    
    def test_invalid_file(self):
        # URL de l'API
        url = "http://127.0.0.1:8000/upload/"
        
        # Chemin du fichier .flac à envoyer
        file_path = "Test\Turdus_merlula.flac"
        
        # Ouvrir le fichier en mode lecture binaire
        with open(file_path, "rb") as file:
            files = {"file": ("audio_file.flac", file, "audio/flac")}
            response = requests.post(url, files=files)
        
        # Vérifier la réponse
        self.assertEqual(response.status_code, 200)
        self.assertIn("error", response.json())
        self.assertEqual(response.json()["error"], "Le fichier doit être un fichier audio .wav ou .mp3")

if __name__ == '__main__':
    unittest.main()