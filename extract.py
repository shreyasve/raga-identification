import os
import librosa
import pandas as pd
import numpy as np
import argparse

def extract_features(audio_file, num_segments=10, output_csv=None):
    columns = [
        'filename', 'chroma_stft', 'spec_cent', 'spec_bw', 'rmse',
        'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6',
        'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12',
        'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20', 'raga'
    ]
    data_list = []

    try:
        y, sr = librosa.load(audio_file, sr=None)
        dur = librosa.get_duration(y=y, sr=sr)

        if dur <= 0:
            print(f"Skipping {audio_file}: duration is too short.")
        else:
            segment_duration = dur / num_segments  # Divide total duration by number of segments
            off = 0
            for i in range(num_segments):
                x, sr = librosa.load(audio_file, sr=None, offset=off, duration=segment_duration)

                chroma_stft = np.mean(librosa.feature.chroma_stft(y=x, sr=sr))
                spec_cent = np.mean(librosa.feature.spectral_centroid(y=x, sr=sr))
                spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=x, sr=sr))
                rmse = np.mean(librosa.feature.rms(y=x))
                mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20)

                mfcc_means = np.mean(mfccs, axis=1)
                data = [
                    os.path.basename(audio_file) + f"-{i+1}", chroma_stft, spec_cent, spec_bw, rmse,
                    *mfcc_means, 'unknown_raga'
                ]

                data_list.append(data)
                off += segment_duration

    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

    dataset = pd.DataFrame(data_list, columns=columns)

    if output_csv:
        dataset.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")

    return dataset

if __name__ == "__main__":
    # Now we expect the audio file path from the command-line argument
    parser = argparse.ArgumentParser(description="Extract audio features from an audio file.")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("-o", "--output_csv", help="Path to save the output CSV file", default=None)

    args = parser.parse_args()

    # Extract features and optionally save to CSV
    features = extract_features(args.audio_file, output_csv=args.output_csv)
    print(features.head())  # Print first few rows of the extracted features for verification