
# Voice Chatbot

### Overview

This Python script implements a voice chatbot that performs the following tasks:

1. Loads an audio file: Converts the audio to a format suitable for processing.
2. Applies Voice Activity Detection (VAD): Filters out non-speech parts of the audio.
3. Transcribes the audio: Converts spoken language into text using the Whisper model.
4. Generates a text response: Uses a pre-trained language model to generate a response based on the transcription.
5. Converts the text to speech (TTS): Generates a spoken version of the text response and saves it as an audio file.

### Dependencies
The script relies on the following libraries:

- **webrtcvad**: For Voice Activity Detection (VAD).
- **numpy**: For numerical operations and handling audio data.
- **pydub**: For audio file manipulation.
- **whisper**: For speech-to-text transcription.
- **torch**: For handling tensors (used with Whisper).
- **transformers**: For text generation using the Hugging Face pipeline.
- **edge_tts**: For text-to-speech conversion using Microsoft's Azure - Cognitive Services.
- **asyncio and nest_asyncio**: For handling asynchronous TTS processing.

Make sure to install these libraries via pip before running the script:
```bash
!pip install -U openai-whisper
!pip install webrtcvad
!pip install pydub
!pip install transformers torch
!pip install edge-tts
!pip install nest_asyncio
```
## Code Breakdown
### 1. Loading the Whisper Model
```bash
WHISPER_MODEL = whisper.load_model(name="base.en")

```
- The script loads a pre-trained **Whisper** model using the whisper library. The **base.en** model is optimized for English transcription.

### 2. Loading the Language Model
```bash
LAMINI_LLM = pipeline("text2text-generation", model="MBZUAI/LaMini-T5-738M")

```
- A text generation pipeline is created using the **transformers** library, loading the **LaMini-T5-738M** model. This model is used to generate text responses based on the transcribed audio input..

### 3. Voice Activity Detection (VAD) Setup
```bash
vad = webrtcvad.Vad()

```
- A VAD object is initialized using the **webrtcvad** library. This object is later used to filter out non-speech portions of the audio.

### 4. Audio File Path and TTS Parameters
```bash
audio_file_path = "/content/sample_query.m4a"  # Audio file path
output_file = "/content/output_speech.mp3"  # Output file path
voice = 'en-US-MichelleNeural' # select any voice
rate = '+10%'  # Speed adjustment (-100% to +100%)
pitch = '-40Hz'  # Pitch adjustment 

```
- The paths for the input audio file and the output TTS file are defined.
- TTS parameters such as voice, speech rate, and pitch are set. You can select different voices and adjust the rate and pitch according to your needs.
- **Male:** 'en-US-GuyNeural', 'en-US-ChristopherNeural', 'en-US-EricNeural', 'en-IN-PrabhatNeural'
- **Female:** 'en-US-AriaNeural', 'en-US-JennyNeural', 'en-IN-NeerjaNeural', 'en-US-MichelleNeural'

### 5. Loading and Preprocessing Audio
```bash
def load_audio(file_path, target_sample_rate=16000):
    ...

```
- **Purpose:** This function loads an audio file and preprocesses it for further analysis.
- **Steps:**
    - Loads the audio using pydub.
    - Converts the audio to mono if it has multiple channels.
    - Resamples the audio to the target sample rate (16 kHz by default).
    - Normalizes the audio to -20 dBFS.
    - Converts the AudioSegment object to a NumPy array of floating-point samples normalized to the range [-1, 1].

### 6. Applying Voice Activity Detection (VAD)
```bash
def apply_vad(audio_samples, sample_rate=16000, vad_threshold=0.5):
    ...

```
- **Purpose:** This function applies VAD to isolate speech segments from the input audio.
- **Steps:**
    - Sets the VAD mode based on the provided threshold (range 0 to 3).
    - Segments the audio into frames of 30 ms each.
    - Checks each frame for speech activity.
    - Collects voiced frames and returns them as a NumPy array.


### 7. Transcribing the Audio
```bash
def transcribe_audio(audio_samples):
    ...

```
- **Purpose:** Converts the processed audio into text using the Whisper model.
- **Steps:**
    - Converts the NumPy array of audio samples into a PyTorch tensor.
    - Passes the tensor to the Whisper model for transcription.
    - Returns the transcription result as text.

### 8. Generating a Text Response
```bash
def generate_response(query):
    ...

```
- **Purpose:** Generates a response to the transcribed text using the LaMini-T5 language model.
- **Steps:**
    - Passes the transcribed text to the text generation model.
    - Extracts and returns the generated text, ensuring that the response is only in two lines.

### 9. Converting Text to Speech (TTS)
```bash
nest_asyncio.apply()
async def text_to_speech(text, output_file=output_file, voice=voice, rate=rate, pitch=pitch):
    ...

```
- **Purpose:** Converts the generated text response to speech using Microsoft's Azure Cognitive Services.
- **Steps:**
    - Initializes the TTS object with the text, voice, rate, and pitch parameters.
    - Uses the **edge_tts** library to synthesize the speech and save it as an audio file.

### 10. Running the Main Function
```bash
def main():
    start_time = time.time()
    ...
if __name__ == "__main__":
    main()

```
- **Purpose:** This is the main function that orchestrates the entire workflow, including loading audio, applying VAD, transcribing audio, generating a response, and converting the response to speech.
- **Steps:**
    - Records the time taken for each step and prints it for performance monitoring.
    - The total execution time is calculated and displayed at the end.

## Performance Metrics
- In the sample run provided, the following times were recorded:

    - Load audio time: 0.2025 seconds
    - VAD processing time: 0.0076 seconds
    - Transcription time: 0.1329 seconds
    - LLM Response time: 5.5618 seconds
    - TTS conversion time: 0.7824 seconds
    - Total execution time: 6.6877 seconds
- The most time-consuming part is generating the response using the language model, which could be optimized for faster response times.

## Conclusion
This script provides a basic framework for building a voice-enabled chatbot that can process audio input, generate intelligent responses, and convert them back to speech. The modular design allows for easy customization and integration with other systems or enhancements.
## 🚀 About Me
My name is Tarpit, and I completed my B.Tech in Computer Engineering with a specialization in AI and Machine Learning in 2024. I'm deeply passionate about harnessing the power of AI to develop innovative solutions and address challenging problems. This project is a testament to my dedication to pushing the boundaries of AI-driven applications, blending state-of-the-art models with practical implementations to create impactful, real-world solutions.


## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/tarpit-patel)


