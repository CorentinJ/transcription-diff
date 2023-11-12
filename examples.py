import logging

import librosa

from transcription_diff.text_diff import transcription_diff, render_text_diff


logging.basicConfig(level="INFO")


audio_fpath = librosa.example("libri2")
wav, sr = librosa.core.load(audio_fpath)

# We'll keep only a short audio to keep the demo concise
# N.B.: without this step we could feed the audio file path directly to transcription_diff()
cut_range = 2.5, 9.5
wav = wav[int(sr * cut_range[0]):int(sr * cut_range[1])]
correct_transcription = \
    "It befell in the month of May, Queen Guenever called her knights of the Table Round and gave them warning."

# # You can listen to the audio using this package, or by playing the file at <audio_fpath>
# import sounddevice as sd
# sd.play(wav, sr, blocking=True)


# Running with all default parameters
diff = transcription_diff(correct_transcription, wav, sr)
print(render_text_diff(diff))

# Providing hints to custom words to whisper has a chance to make it transcribe that word
diff = transcription_diff(correct_transcription, wav, sr, custom_words=["Guenever"])
print(render_text_diff(diff))

# Increase the model size generally increases ASR accuracy
diff = transcription_diff(correct_transcription, wav, sr, custom_words=["Guenever"], whisper_model_size=3)
print(render_text_diff(diff))
