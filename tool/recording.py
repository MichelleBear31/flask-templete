import os
import pyaudio
import librosa
import wave
def record_audio_to_file(filename, duration=1, channels=1, rate=44100, frames_per_buffer=1):
    """Record user's input audio and save it to the specified file."""
    FORMAT = pyaudio.paInt16
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=FORMAT, channels=channels, rate=rate, frames_per_buffer=frames_per_buffer, input=True)
    print("Start recording...")

    frames = []
    # Record for the given duration
    for _ in range(0, int(rate / frames_per_buffer * duration)):
        data = stream.read(frames_per_buffer)
        frames.append(data)
    print("Recording stopped")
    stream.stop_stream()
    stream.close()
    p.terminate()
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio recorded and saved to {filename}")
    return filename
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename=os.path.join(current_dir, 'static','audio','user_input.wav')
    record_audio_to_file(filename)