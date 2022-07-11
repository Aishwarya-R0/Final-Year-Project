import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import shutil

def main():
    freq = 44100

        # Recording duration
    duration = 5

        # Start recorder with the given values
        # of duration and sample frequency
    recording = sd.rec(int(duration * freq),
                           samplerate=freq, channels=2)
    print('Speak Now')
        # Record audio for the given number of seconds
    sd.wait()

    print('Recorded')
        # Convert the NumPy array to audio file
    wv.write("recording1.wav", recording, freq, sampwidth=2)
    src_path = r"C:/Users/Dell/OneDrive/Documents/EndUser/recording1.wav"
    dst_path = r"C:/Users/Dell/OneDrive/Documents/sharedfolder/recording1.wav"
    shutil.move(src_path, dst_path)

if __name__=="__main__":
    main()