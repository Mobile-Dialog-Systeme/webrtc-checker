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

# Browser Recording Implementation (SoSci Survey)

In addition to the offline audio analysis tools, this repository provides a **browser-based speech recording implementation** used for collecting speech data in online surveys.

The implementation is designed for **SoSci Survey** and uses the **MediaStream Recorder API** to capture microphone input directly in the participant’s browser.

The recording configuration intentionally **disables all browser-side audio processing** in order to minimise artefacts and preserve the raw microphone signal.

Disabled processing modules:

- echo cancellation
- noise suppression
- automatic gain control (AGC)

These processing steps are often designed for telecommunication scenarios (e.g. video calls) and may introduce **signal distortions such as amplitude jumps, clipping, or temporal artefacts** when used for speech research.

---

# Recording Format

The implementation records audio in:
audio/ WEBM


Using the **WebM container with Opus codec**.

In our evaluation, this format produced **fewer clicking artefacts** than recordings exported as `.wav` or `.mp3` using native WebRTC implementations.

---

# Integration in SoSci Survey

The recording interface can be implemented in **SoSci Survey** using the question type:
Transfer file contents/ Datei-Inhalte übertragen


### Steps

1. Create a new question in SoSci Survey.
2. Select the question type **"Transfer file contents"**.
3. Insert the provided HTML/JavaScript code into the question editor.
4. The recording is transmitted to SoSci Survey using the internal function: %q.id%.sendBLOB(blob)

This stores the recorded audio file together with the participant's response data.

---

# Recording Workflow

The interface provides:

- Microphone selection
- Recording start/stop controls
- Recording duration timer
- Level meter for participant feedback
- Audio playback for recording verification

### Workflow for participants

1. Select microphone device (if multiple devices exist)
2. Press **START**
3. Speak the required speech material
4. Press **STOP**
5. Listen to the recording
6. Continue the survey if the recording is acceptable

---

# Why All Processing Is Disabled

Many browsers apply **automatic signal processing** during recording.

While useful for communication applications, these algorithms can introduce artefacts such as:

- amplitude fluctuations
- clipping
- transient clicks
- dynamic compression

For phonetic or speech science research, such alterations can distort acoustic measurements.

Therefore the following configuration is used:

```javascript
audio: {
  echoCancellation: false,
  noiseSuppression: false,
  autoGainControl: false,
  channelCount: 1
}

This captures the raw microphone signal with minimal browser-side modification.

# Research Context

The recording configuration used in this repository was evaluated in a comparative study analysing artefact distributions across three browser recording frameworks:
-WebRTC
-RecordRTC
-MediaStream Recorder

The evaluation showed that **MediaStream Recorder with all processing disabled** produced the **lowest number of severe artefacts** such as clipping and clicking.

# Citation

If you use this code, the SoSci Survey recording implementation, or the audio artefact analysis methods in your research, please cite the following publication:

**Hacker, A., Bakker, I. S., & Siegert, I.**  
*Evaluation of WebRTC as a Framework for Voice Recordings in Online Surveys.*  
Mobile Dialog Systems, Otto von Guericke University Magdeburg &  
Research Group on Intelligent Assistive Systems for Psychotherapy, University Hospital Magdeburg.

Authors:  
- Anabell Hacker¹²  
- Iris Sidonie Bakker²  
- Ingo Siegert¹²  

¹ Mobile Dialog Systems, Otto von Guericke University Magdeburg  
² Research Group on Intelligent Assistive Systems for Psychotherapy, University Hospital Magdeburg  

Contact:  
anabell.hacker@ovgu.de  
ingo.siegert@ovgu.de

---

# BibTeX

You can cite the paper using the following BibTeX entry:

```bibtex
@inproceedings{Hacker2026webrtc,
  title={Evaluation of WebRTC as a Framework for Voice Recordings in Online Surveys},
  author={Hacker, Anabell and Bakker, Iris Sidonie and Siegert, Ingo},
  booktitle={Tagungsband der 37. Konferenz Elektronische Sprachsignalverarbeitung ESSV},
  year={2026},
  pages={200--207}
}
