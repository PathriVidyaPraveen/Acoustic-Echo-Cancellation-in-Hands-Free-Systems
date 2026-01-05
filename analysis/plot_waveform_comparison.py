import numpy as np
import matplotlib.pyplot as plt
import os
import config.aec_config as cfg
from core.overlap_save import OverlapSave
from core.pbfdaf import PBFDAF
from core.adaptive_update import pbfda_nlms_update
from nlp.smoothing import GainSmoother
from simulation.generate_signals import generate_test_signals


def main():

    results_dir = "results"
    os.makedirs(results_dir,exist_ok=True)

    # generate echo only signal
    far_end,_,_,echo_component = generate_test_signals(duration_sec=5.0,fs=cfg.SAMPLE_RATE)

    mic_signal = echo_component.copy()

    num_blocks = len(far_end)//cfg.BLOCK_SIZE

    os_far = OverlapSave(cfg.BLOCK_SIZE,cfg.FFT_SIZE)
    os_mic = OverlapSave(cfg.BLOCK_SIZE,cfg.FFT_SIZE)

    pbfdaf = PBFDAF(cfg.NUM_PARTITIONS,cfg.FFT_SIZE)

    num_bins = cfg.FFT_SIZE//2 + 1
    gain_smoother = GainSmoother(num_bins,alpha=0.7)

    output_signal = np.zeros_like(mic_signal)

    print("Running waveform comparison...")

    for i in range(num_blocks):

        idx0 = i*cfg.BLOCK_SIZE
        idx1 = idx0+cfg.BLOCK_SIZE

        x_block = far_end[idx0:idx1]
        d_block = mic_signal[idx0:idx1]

        x_buf = os_far.process(x_block)
        d_buf = os_mic.process(d_block)

        X_f = np.fft.rfft(x_buf)
        pbfdaf.update_input_history(X_f)

        Y_hat_f = pbfdaf.estimate_echo()
        y_hat_time = np.fft.irfft(Y_hat_f,n=cfg.FFT_SIZE)

        valid_start = cfg.FFT_SIZE-cfg.BLOCK_SIZE
        y_hat_block = y_hat_time[valid_start:]

        e_block = d_block-y_hat_block

        e_full = d_buf-y_hat_time
        E_f = np.fft.rfft(e_full)

        pbfda_nlms_update(
            pbfdaf.W,
            pbfdaf.X_history,
            E_f,
            step_size=cfg.STEP_SIZE,
            epsilon=cfg.EPSILON,
            adapt=True
        )
        mag_E = np.abs(E_f)
        mag_Y = np.abs(Y_hat_f)
        est_residual = cfg.NLP_AGGRESSIVENESS*mag_Y

        raw_gain = (mag_E - est_residual)/(mag_E + 1e-10)
        raw_gain = np.clip(raw_gain,cfg.NLP_MIN_GAIN_LINEAR,1.0)

        smooth_gain = gain_smoother.smooth(raw_gain)
        E_enhanced_f = E_f*smooth_gain

        e_time = np.fft.irfft(E_enhanced_f,n=cfg.FFT_SIZE)
        output_signal[idx0:idx1] = e_time[valid_start:]

    start_sec = 1.0
    dur_sec = 0.03

    start = int(start_sec*cfg.SAMPLE_RATE)
    end = start + int(dur_sec*cfg.SAMPLE_RATE)

    t = np.arange(start, end)/cfg.SAMPLE_RATE

    plt.figure(figsize=(12,6))
    plt.plot(t, mic_signal[start:end],label="Mic Signal (Echo)",alpha=0.6)
    plt.plot(t, output_signal[start:end],label="After AEC",linewidth=2)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Waveform Comparison: Before vs After Echo Cancellation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(results_dir,"waveform_comparison.png")
    plt.savefig(save_path,dpi=200)
    plt.show()

    print(f"Saved waveform plot to {save_path}")


if __name__ == "__main__":
    main()
