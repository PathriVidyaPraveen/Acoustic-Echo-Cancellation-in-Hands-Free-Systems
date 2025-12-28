import numpy as np
import matplotlib.pyplot as plt
import os

import config.aec_config as cfg
from core.overlap_save import OverlapSave
from core.pbfdaf import PBFDAF
from core.adaptive_update import pbfda_nlms_update
from dtd.coherence_dtd import CoherenceDTD
from nlp.smoothing import GainSmoother
from simulation.generate_signals import generate_test_signals


def main():

    results_dir = "results"
    os.makedirs(results_dir,exist_ok=True)

    # generate test singals
    far_end, mic_signal,_,_ = generate_test_signals(duration_sec=5.0,fs=cfg.SAMPLE_RATE)

    num_blocks = len(far_end)//cfg.BLOCK_SIZE

    # initialization
    os_far_end = OverlapSave(cfg.BLOCK_SIZE,cfg.FFT_SIZE)
    os_mic = OverlapSave(cfg.BLOCK_SIZE,cfg.FFT_SIZE)

    pbfdaf = PBFDAF(cfg.NUM_PARTITIONS,cfg.FFT_SIZE)
    dtd = CoherenceDTD(cfg.DTD_COHERENCE_THRESHOLD,cfg.DTD_SMOOTHING_FACTOR)

    num_bins = cfg.FFT_SIZE//2 + 1
    gain_smoother = GainSmoother(num_bins=num_bins,alpha=0.7)

    raw_gain_log = []
    smooth_gain_log = []

    print("Running NLP gain analysis...")

    for i in range(num_blocks):

        idx_start = i*cfg.BLOCK_SIZE
        idx_end = idx_start+cfg.BLOCK_SIZE

        x_block = far_end[idx_start:idx_end]
        d_block = mic_signal[idx_start:idx_end]

        x_buf = os_far_end.process(x_block)
        d_buf = os_mic.process(d_block)

        X_f = np.fft.rfft(x_buf)
        D_f = np.fft.rfft(d_buf)

        is_double_talk = dtd.detect(X_f,D_f)
        pbfdaf.update_input_history(X_f)
        Y_hat_f = pbfdaf.estimate_echo()

        # Time domain error
        y_hat_time = np.fft.irfft(Y_hat_f,n=cfg.FFT_SIZE)
        valid_start = cfg.FFT_SIZE -cfg.BLOCK_SIZE
        y_hat_block = y_hat_time[valid_start:]
        e_block = d_block - y_hat_block

        e_padded = np.concatenate((np.zeros(cfg.BLOCK_SIZE), e_block))
        E_f = np.fft.rfft(e_padded,n=cfg.FFT_SIZE)

        # Adaptive update
        pbfda_nlms_update(
            W=pbfdaf.W,
            X_history=pbfdaf.X_history,
            E=E_f,
            step_size=cfg.STEP_SIZE,
            epsilon=cfg.EPSILON,
            adapt=not is_double_talk
        )

        # NLP gain computed
        epsilon = 1e-10
        magnitude_E = np.abs(E_f)
        magnitude_Y = np.abs(Y_hat_f)
        est_residual = cfg.NLP_AGGRESSIVENESS * magnitude_Y
        raw_gain = (magnitude_E - est_residual)/(magnitude_E + epsilon)
        raw_gain = np.clip(raw_gain,cfg.NLP_MIN_GAIN_LINEAR,1.0)
        smooth_gain = gain_smoother.smooth(raw_gain)

        raw_gain_log.append(raw_gain.copy())
        smooth_gain_log.append(smooth_gain.copy())

    raw_gain_log = np.array(raw_gain_log)
    smooth_gain_log = np.array(smooth_gain_log)
    time_axis = np.arange(raw_gain_log.shape[0])
    bin_idx = num_bins//4

    # single frequency bin
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis,raw_gain_log[:,bin_idx],label="Raw Gain",alpha=0.5)
    plt.plot(time_axis,smooth_gain_log[:,bin_idx],label="Smoothed Gain",linewidth=2)
    plt.xlabel("Block Index")
    plt.ylabel("Gain")
    plt.title(f"NLP Gain - Frequency Bin {bin_idx}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    single_bin_path = os.path.join(results_dir,"nlp_gain_single_bin.png")
    plt.savefig(single_bin_path, dpi=200)
    plt.show()

    # average gain over all bins
    avg_raw = np.mean(raw_gain_log, axis=1)
    avg_smooth = np.mean(smooth_gain_log, axis=1)

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis,avg_raw,label="Avg Raw Gain",alpha=0.5)
    plt.plot(time_axis,avg_smooth,label="Avg Smoothed Gain",linewidth=2)
    plt.xlabel("Block Index")
    plt.ylabel("Average Gain")
    plt.title("Average NLP Gain Over Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    avg_path = os.path.join(results_dir,"nlp_gain_average.png")
    plt.savefig(avg_path, dpi=200)
    plt.show()
    print("Gain analysis complete.")
    print(f"Saved plots to: {results_dir}/")
    
if __name__ == "__main__":
    main()
