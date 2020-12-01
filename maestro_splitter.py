import pandas as pd
import random

'''
This file creates subsets within the predefined train/val/split of the
MAESTRO dataset.

I got the following statistics:
    Number of train performances: 125
    Duration of training: 20.0 hours
    Number of validation performances: 13
    Duration of validation: 2.5 hours
    Number of test performances: 23
    Duration of testing: 2.6 hours

'''

desired_gb = 15

total_duration_hours = 201.2/121.8*desired_gb
train_seconds = total_duration_hours*0.8*60*60
val_seconds = total_duration_hours*0.1*60*60
test_seconds = val_seconds

maestro_df = pd.read_csv('maestro-v2.0.0/maestro-v2.0.0.csv')
train_data = maestro_df[maestro_df['split'] == 'train']
val_data = maestro_df[maestro_df['split'] == 'validation']
test_data = maestro_df[maestro_df['split'] == 'test']

# Create a balanced subset for each split so each composer has the same amount
# of time in it

# Get total duration for each composer within each split
train_min_composer = train_data.groupby('canonical_composer').agg('sum').reset_index()
val_min_composer = val_data.groupby('canonical_composer').agg('sum').reset_index()
test_min_composer = test_data.groupby('canonical_composer').agg('sum').reset_index()

# Get composers in each split and randomize the order
train_composers = train_min_composer['canonical_composer'].tolist()
##print(train_composers)
random.shuffle(train_composers)
val_composers = val_min_composer['canonical_composer'].tolist()
##print(val_composers)
random.shuffle(val_composers)
test_composers = test_min_composer['canonical_composer'].tolist()
##print(test_composers)
random.shuffle(test_composers)

# Get all of the midi files belonging to each composer in each split
train_songs = {}
for composer in train_composers:
    songs = train_data[train_data['canonical_composer'] == composer]
    train_songs[composer] = songs['midi_filename'].tolist()
val_songs = {}
for composer in val_composers:
    songs = val_data[val_data['canonical_composer'] == composer]
    val_songs[composer] = songs['midi_filename'].tolist()
test_songs = {}
for composer in test_composers:
    songs = test_data[test_data['canonical_composer'] == composer]
    test_songs[composer] = songs['midi_filename'].tolist()

# Iterate through each list and pick one song at a time
duration = 0
i = 0
train_songs_subset = []
while duration < train_seconds:
    composer = train_composers[i%len(train_composers)]
    songs = train_songs[composer]
    if len(songs) == 0:
        i += 1
        continue
    random.shuffle(songs)
    picked_song = songs[0]
    songs.pop(0)
    train_songs[composer] = songs
    song_data = train_data[train_data['midi_filename'] == picked_song]
    duration += song_data['duration'].tolist()[0]
    train_songs_subset.append(picked_song)
    i += 1
print('Number of Performances in Training:', len(train_songs_subset))
print('Total Duration in Training:', duration)
duration = 0
i = 0
val_songs_subset = []
while duration < val_seconds:
    composer = val_composers[i%len(val_composers)]
    songs = val_songs[composer]
    if len(songs) == 0:
        i += 1
        continue
    random.shuffle(songs)
    picked_song = songs[0]
    songs.pop(0)
    val_songs[composer] = songs
    song_data = val_data[val_data['midi_filename'] == picked_song]
    duration += song_data['duration'].tolist()[0]
    val_songs_subset.append(picked_song)
    i += 1
print('Number of Performances in Validation:', len(val_songs_subset))
print('Total Duration in Validation:', duration)
duration = 0
i = 0
test_songs_subset = []
while duration < test_seconds:
    composer = test_composers[i%len(test_composers)]
    songs = test_songs[composer]
    if len(songs) == 0:
        i += 1
        continue
    random.shuffle(songs)
    picked_song = songs[0]
    songs.pop(0)
    test_songs[composer] = songs
    song_data = test_data[test_data['midi_filename'] == picked_song]
    duration += song_data['duration'].tolist()[0]
    test_songs_subset.append(picked_song)
    i += 1
print('Number of Performances in Testing:', len(test_songs_subset))
print('Total Duration in Testing:', duration)

# Generate a pandas dataframe and write it out as a csv file for each split
train_subset_df = train_data[train_data['midi_filename'].isin(train_songs_subset)]
train_subset_df.to_csv('train_midi_df.csv', index=False)
val_subset_df = val_data[val_data['midi_filename'].isin(val_songs_subset)]
val_subset_df.to_csv('val_midi_df.csv', index=False)
test_subset_df = test_data[test_data['midi_filename'].isin(test_songs_subset)]
test_subset_df.to_csv('test_midi_df.csv', index=False)
