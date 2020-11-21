# This file adjusts the speed of a WAV file to match that of its corresponding
# MIDI file's

import librosa

old_path = 'Love_Story_meets_Viva_la_Vida.wav' # WAV file to adjust
new_path = 'Adjusted_Love_Story_meets_Viva_la_Vida.wav' # output WAV file
tempo = 100 # adjust WAV file's tempo to this tempo

# Estimate tempo of song
# This can be inaccurate and it is better to estimate the tempo yourself
song, sr = librosa.load(old_path) # load audio data, sampling rate
est_tempo = librosa.beat.tempo(y=song)[0]
print('Estimated tempo of song:', est_tempo)

# Time stretch audio file so its tempo matches the MIDI file's
new_song = librosa.effects.time_stretch(song, tempo/est_tempo)
librosa.output.write_wav(new_path, new_song, sr=sr)
