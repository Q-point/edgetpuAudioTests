import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_magnitude_and_phase(wav_file):
    # Load the audio file
    y, sr = librosa.load(wav_file)

    # Compute the Short-Time Fourier Transform
    stft_result = librosa.stft(y)
    magnitude, phase = librosa.magphase(stft_result)
    log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)

    # Plot the magnitude spectrogram
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    librosa.display.specshow(log_magnitude, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log Magnitude Spectrogram')

    # Plot the phase spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(librosa.amplitude_to_db(phase), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log Phase Spectrogram')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":   
    wav_file_path = '0_03_1.wav'
    plot_magnitude_and_phase(wav_file_path)
