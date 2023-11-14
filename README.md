# transcription-diff
A small python library to find differences between audio and transcriptions

Example (audio as mp4 to allow an embed):

https://github.com/CorentinJ/transcription-diff/assets/12038136/41fda0a8-92bb-46fe-a7b7-903ccfed3463

```python
from transcription_diff.text_diff import transcription_diff, render_text_diff

diff = transcription_diff("You can go pretty far in life if you're a perfect sphere in a vacuum", "sphere.mp4")
print(render_text_diff(diff))
```

```diff
! Well
You can go pretty far in life
! when
+ if
you're a perfect sphere in a vacuum
```

### Mechanism
- The library relies on [openai-whisper](https://github.com/openai/whisper) to perform Audio Speech Recognition unguided by the transcription
- It then compares the expected transcription to the output of Whisper, ignoring superfluous characters
- It returns the output in a simple structure, keeping the original text format of the transcription

### Limitations
- Only a single hypothesis is considered for the ASR output, leaving the possibility of missing a hypothesis that would satisfy the expected transcription
- The ASR output is not in the phoneme space, making homophones prone to false positives
- Rare words unknown to Whisper require to be explicitly passed to the function, and have no guarantee of being properly recognized by Whisper
- Currently only annotates up to 30 seconds of audio per sample

## Installation
`pip install transcription-diff`

## Short term TODOs
- [ ] Phoneme-level comparison
- [ ] User handling of model cache
- [ ] Support for audios longer than 30s

## Long shot TODOs
- [ ] More robust support for non-English languages
- [ ] Inverse normalization support for less false positives
