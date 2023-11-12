import librosa

from transcription_diff.text_diff import transcription_diff, render_text_diff

audio_fpath = librosa.example("libri2")
wav, sr = librosa.core.load(audio_fpath)

# We'll keep only a short audio to keep the demo concise
cut_range = 2.5, 5.25
wav = wav[int(sr * cut_range[0]):int(sr * cut_range[1])]
correct_transcription = "It befell in the month of May, Queen Guenever called to her knights of the Table Round"

# # You can listen to the audio using this package, or by playing the file at <audio_fpath>
# import sounddevice as sd
# sd.play(wav, sr, blocking=True)

print(render_text_diff(transcription_diff(correct_transcription, wav, sr)))
