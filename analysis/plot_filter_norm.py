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

    far_end, _,_,echo_component = generate_test_signals(duration_sec=5.0,fs=cfg.SAMPLE_RATE)
    mic_signal = echo_component.copy()

    num_blocks = len(far_end)//cfg.BLOCK_SIZE
    os_far = OverlapSave(cfg.BLOCK_SIZE,cfg.FFT_SIZE)
    os_mic = OverlapSave(cfg.BLOCK_SIZE,cfg.FFT_SIZE)
    pbfdaf = PBFDAF(cfg.NUM_PARTITIONS,cfg.FFT_SIZE)

    filter_norm = []

    print("Running filter norm analysis...")

    for i in range(num_blocks):

        idx0 = i * cfg.BLOCK_SIZE
        idx1 = idx0 + cfg.BLOCK_SIZE

        x_block = far_end[idx0:idx1]
        d_block = mic_signal[idx0:idx1]
        x_buf = os_far.process(x_block)
        d_buf = os_mic.process(d_block)

        X_f = np.fft.rfft(x_buf)
        D_f = np.fft.rfft(d_buf)


        pbfdaf.update_input_history(X_f)

        Y_hat_f = pbfdaf.estimate_echo()
        y_hat_time = np.fft.irfft(Y_hat_f,n=cfg.FFT_SIZE)
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

        W_norm = np.linalg.norm(pbfdaf.W)
        filter_norm.append(W_norm)

    filter_norm = np.array(filter_norm)

    time_axis = (np.arange(len(filter_norm)) *cfg.BLOCK_SIZE / cfg.SAMPLE_RATE)


    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, filter_norm, linewidth=2)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Filter Norm ||W||")
    plt.title("Adaptive Filter Norm vs Time")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(results_dir,"filter_norm_vs_time.png")
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"Saved filter norm plot to {save_path}")


if __name__ == "__main__":
    main()
