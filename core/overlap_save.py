import numpy as np

class OverlapSave:
    # implement overlap save bufferring logic for block based freq domain processing

    def __init__(self,block_size,fft_size): # initialize buffering system
        if fft_size < block_size:
            raise ValueError(f"FFT size {fft_size} must be >= block size {block_size}")
        self.block_size = block_size
        self.fft_size = fft_size
        self.buffer = np.zeros(self.fft_size,dtype=np.float32)

    def process(self,x_block):
        # update block with new block of samplees
        if len(x_block)!= self.block_size:
            raise ValueError(f"Input block length {len(x_block)} does not match configured size {self.block_size}")
        self.buffer = np.roll(self.buffer ,-self.block_size)

        self.buffer[-self.block_size:] = x_block
        return self.buffer

        
