import pyaudio
import wave
import audioop

chunk = 1024
filename="sound/asian/assembly_line.wav"
wf = wave.open(filename, 'rb')

# Create an interface to PortAudio
p = pyaudio.PyAudio()

# Open a .Stream object to write the WAV file to
# 'output = True' indicates that the sound will be played rather than recorded
stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = wf.getframerate(),
                output = True)

# Read data in chunks
data = wf.readframes(chunk)

# Play the sound by writing the audio data to the stream
while data != b'':
    stream.write(data)
    data = wf.readframes(chunk)
    rms= audioop.rms(data,2)
    if rms>500:
        print(rms)

# Close and terminate the stream
stream.close()
p.terminate()
