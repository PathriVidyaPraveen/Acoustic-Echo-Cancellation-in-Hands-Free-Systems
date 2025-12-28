import torchaudio
out = torchaudio.datasets.LIBRISPEECH(".", url="dev-clean", download=True)
waveform, sr, _, _, _, _ = out[0]
torchaudio.save("sample.wav", waveform, sr)
