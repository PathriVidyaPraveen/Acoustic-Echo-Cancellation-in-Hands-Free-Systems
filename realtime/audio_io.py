import sounddevice as sd
import numpy as np
import queue

class AudioIO:
    # handle real time audio IO for AEC system

    def __init__(self,sample_rate,block_size,input_device=None,output_device=None):
        self.fs = sample_rate
        self.block_size = block_size
        self.input_device = input_device
        self.output_device = output_device
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.stream = None
        self.active = False

    def _callback(self,in_data,out_data,frames,time,status):
        # PortAudio/SoundDevice callback

        if status:
            print(f"Audio Callback Status: {status}")

        self.input_queue.put(in_data.copy().flatten())
        try:
            data = self.output_queue.get_nowait()
        except queue.Empty:
            data = np.zeros(frames,dtype=np.float32)

        out_data[:] = data.reshape(-1,1)

    def start(self):
        # open stream and start audio processing
        self.active = True

        for _ in range(2):
            self.output_queue.put(np.zeros(self.block_size,dtype=np.float32))
            
        self.stream= sd.Stream(
            samplerate=self.fs,
            blocksize=self.block_size,
            device=(self.input_device,self.output_device),
            channels=1,
            dtype=np.float32,
            latency='low',
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        # stops audio stream and closes resources
        self.active=False
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def read(self):
        # blocking read - wait for nexct block of microphone data

        return self.input_queue.get()
    
    def write(self,block):
        # queues block of audio to be played by speakers
        if len(block) != self.block_size:
            raise ValueError(f"Block size {len(block)} mismatch- Expected {self.block_size}")
        
        self.output_queue.put(block.astype(np.float32))


