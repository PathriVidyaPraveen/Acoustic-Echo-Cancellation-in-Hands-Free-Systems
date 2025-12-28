import numpy as np
import matplotlib.pyplot as plt
import os
import config.aec_config as cfg
from core.overlap_save import OverlapSave
from core.pbfdaf import PBFDAF
from core.adaptive_update import pbfda_nlms_update
from simulation.generate_signals import generate_test_signals


def main():


    results_dir = "results"
    os.makedirs(results_dir,exist_ok=True)

    # generate test singals with far end only
    far_end,mic_signal,_,echo_component = generate_test_signals(duration_sec=5.0,fs=cfg.SAMPLE_RATE)
    mic_signal = echo_component.copy()
    num_blocks = len(far_end)//cfg.BLOCK_SIZE
    os_far = OverlapSave(cfg.BLOCK_SIZE,cfg.FFT_SIZE)
    os_mic = OverlapSave(cfg.BLOCK_SIZE,cfg.FFT_SIZE)

    pbfdaf = PBFDAF(cfg.NUM_PARTITIONS,cfg.FFT_SIZE)

    erle_per_block = []

    print("Running ERLE vs Time analysis...")

    for i in range(num_blocks):

        idx0 = i*cfg.BLOCK_SIZE
        idx1 = idx0 + cfg.BLOCK_SIZE

        x_block = far_end[idx0:idx1]
        d_block = mic_signal[idx0:idx1]
        x_buf = os_far.process(x_block)
        d_buf = os_mic.process(d_block)

        X_f = np.fft.rfft(x_buf)
        D_f = np.fft.rfft(d_buf)

        pbfdaf.update_input_history(X_f)

        # Echo estimate
        Y_hat_f = pbfdaf.estimate_echo()
        y_hat_time = np.fft.irfft(Y_hat_f,n=cfg.FFT_SIZE)

        valid_start = cfg.FFT_SIZE -cfg.BLOCK_SIZE
        y_hat_block = y_hat_time[valid_start:]

        e_block = d_block - y_hat_block

        # frequency-domain error for adaptation
        e_full = d_buf - y_hat_time
        E_f = np.fft.rfft(e_full)

        pbfda_nlms_update(
            pbfdaf.W,
            pbfdaf.X_history,
            E_f,
            step_size=cfg.STEP_SIZE,
            epsilon=cfg.EPSILON,
            adapt=True
        )

        echo_power = np.mean(d_block** 2) + 1e-10
        error_power = np.mean(e_block**2) + 1e-10

        erle_db = 10*np.log10(echo_power/error_power)
        erle_per_block.append(erle_db)

    erle_per_block = np.array(erle_per_block)

    # smooth ERLE
    window = 10
    erle_smooth = np.convolve(erle_per_block,np.ones(window)/ window,mode="same")

    time_axis = (np.arange(len(erle_smooth))*cfg.BLOCK_SIZE /cfg.SAMPLE_RATE)

    plt.figure(figsize=(12, 6))
    plt.plot(time_axis,erle_smooth,linewidth=2)
    plt.xlabel("Time (seconds)")
    plt.ylabel("ERLE (dB)")
    plt.title("ERLE vs Time (Far-End Only)")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(results_dir,"erle_vs_time.png")
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"Saved ERLE plot to {save_path}")


if __name__ == "__main__":
    main()
