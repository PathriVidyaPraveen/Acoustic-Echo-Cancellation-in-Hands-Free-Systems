import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import numpy as np
import config.aec_config as cfg
from realtime.audio_io import AudioIO
from core.overlap_save import OverlapSave
from core.pbfdaf import PBFDAF
from core.adaptive_update import pbfda_nlms_update
from dtd.coherence_dtd import CoherenceDTD
from nlp.smoothing import GainSmoother


def main():

    print("Starting Real-Time AEC...")
    # audio IO

    audio = AudioIO(sample_rate=cfg.SAMPLE_RATE,block_size=cfg.BLOCK_SIZE)

   # DSP blocks
    os_mic = OverlapSave(cfg.BLOCK_SIZE,cfg.FFT_SIZE)
    pbfdaf = PBFDAF(cfg.NUM_PARTITIONS,cfg.FFT_SIZE)

    dtd = CoherenceDTD(threshold=cfg.DTD_COHERENCE_THRESHOLD,smoothing=cfg.DTD_SMOOTHING_FACTOR)

    num_bins = cfg.FFT_SIZE//2 + 1
    gain_smoother = GainSmoother(num_bins=num_bins,alpha=0.8)


    x_ref_buffer = np.zeros(cfg.BLOCK_SIZE, dtype=np.float32)

    audio.start()
    print("AEC running....Press Ctrl+C to stop.")

    try:
        while True:
            # read mic input
            d_block = audio.read()
            # reference signal
            x_block = x_ref_buffer.copy()
            # FFT processing
            d_buf = os_mic.process(d_block)
            X_f = np.fft.rfft(np.concatenate((np.zeros(cfg.FFT_SIZE -cfg.BLOCK_SIZE),x_block)))
            D_f = np.fft.rfft(d_buf)

            # double talk detection
            is_double_talk = dtd.detect(X_f,D_f)

            # echo estimation
            pbfdaf.update_input_history(X_f)
            Y_hat_f = pbfdaf.estimate_echo()

            y_hat_time = np.fft.irfft(Y_hat_f,n=cfg.FFT_SIZE)

            valid_start = cfg.FFT_SIZE -cfg.BLOCK_SIZE
            y_hat_block = y_hat_time[valid_start:]

            # error signal
            e_block = d_block - y_hat_block
            e_full = d_buf - y_hat_time
            E_f = np.fft.rfft(e_full)

            # adaptive update
            pbfda_nlms_update(
                pbfdaf.W,
                pbfdaf.X_history,
                E_f,
                step_size=0.08,
                epsilon=cfg.EPSILON,
                adapt=False
            )

            # NLP
            magnitude_E = np.abs(E_f)
            magnitude_Y = np.abs(Y_hat_f)

            est_residual = 1.2 * magnitude_Y
            raw_gain = np.ones_like(magnitude_E)

            valid = magnitude_E > 1e-4
            raw_gain[valid] = (magnitude_E[valid]-est_residual[valid])/magnitude_E[valid]

            raw_gain = np.clip(raw_gain,0.1,1.0)

            smooth_gain = gain_smoother.smooth(raw_gain)

            E_enhanced_f = E_f*smooth_gain
            e_enhanced_time = np.fft.irfft(E_enhanced_f,n=cfg.FFT_SIZE)

            e_out = e_enhanced_time[valid_start:]
            e_out = np.clip(e_out, -0.9, 0.9)


            audio.write(e_out)
            x_ref_buffer = e_out.copy()

    except KeyboardInterrupt:
        print("\nStopping AEC...")
        audio.stop()


if __name__ == "__main__":
    main()
