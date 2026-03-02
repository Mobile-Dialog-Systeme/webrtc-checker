# Audio Quality Analysis

This project analyzes audio files for a variety of quality issues, including:

- Volume jumps
- barley hearable passages resulting from volume jumps
- Clipping
- Clicks
- Reverberation or rough sound
- Noise detection

It processes audio files in a folder (including subfolders) and generates:

1. **Overview plots** showing detected events.
2. **Summary Excel file** with file names and detected events.

---

## Features

- Detects **no or low Signal**.
- Detects **volume jumps** using theshold.
- Detects **very low volume** using RMS thresholding combined with voicing probability and spectral features.
- Detects **clicks** (different types, multiple methods).
- Detects **clipping** (overload) events.
- Detects **reverberation / rough sound** patterns.
- Detects general **noise** in audio.

- Generates **visual overview plots** for each file.
- Generates a **summary Excel** with all events.

---

## Requirements

- Python 3.10+
- Libraries:
  ```bash
  pip install librosa numpy scipy pandas matplotlib 
  
## Usage / Setup

1. **Prepare your audio files**  
   - Place all audio files (wav, mp3, ogg) in a folder.  
   - If you have subfolders inside that folder, the script can scan them too. Choose #solution 2: audiofiles in subfolders in a folder. And comment out solution 1.

2. **Set the paths in the script**  
   Open the main Python script add your folder paths