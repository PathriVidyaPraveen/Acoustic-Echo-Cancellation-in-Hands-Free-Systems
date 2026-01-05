import numpy as np
import matplotlib.pyplot as plt
import os
import config.aec_config as cfg
from core.overlap_save import OverlapSave
from dtd.coherence_dtd import CoherenceDTD


def main():

    results_dir = "results"
    os.makedirs(results_dir,exist_ok=True)

    fs = cfg.SAMPLE_RATE
    block = cfg.BLOCK_SIZE
    fft_size = cfg.FFT_SIZE

    duration = 5.0
    total_samples = int(duration*fs)

    np.random.seed(0)

    far_end = np.random.randn(total_samples).astype(np.float32)

    near_end = np.zeros_like(far_end)
    dt_start = int(2.0*fs)
    dt_end = int(3.0*fs)
    near_end[dt_start:dt_end] = 0.8*np.random.randn(dt_end-dt_start)

    mic_signal = far_end+near_end

    
    os_far = OverlapSave(block,fft_size)
    os_mic = OverlapSave(block,fft_size)

    dtd = CoherenceDTD(threshold=cfg.DTD_COHERENCE_THRESHOLD,smoothing=cfg.DTD_SMOOTHING_FACTOR)

    coherence_log = []
    dtd_log = []

    num_blocks = total_samples//block

    print("Running DTD behavior analysis...")
    for i in range(num_blocks):

        idx0 = i*block
        idx1 = idx0+block

        x_block = far_end[idx0:idx1]
        d_block = mic_signal[idx0:idx1]

        x_buf = os_far.process(x_block)
        d_buf = os_mic.process(d_block)

        X_f = np.fft.rfft(x_buf)
        D_f = np.fft.rfft(d_buf)

        is_double_talk = dtd.detect(X_f, D_f)

        # compute average coherence for plotting
        Pxx = dtd.Pxx
        Pdd = dtd.Pdd
        Pxd = dtd.Pxd

        coherence = np.abs(Pxd)** 2/(Pxx*Pdd+1e-10)
        avg_coh = np.mean(coherence)

        coherence_log.append(avg_coh)
        dtd_log.append(int(is_double_talk))

    coherence_log = np.array(coherence_log)
    dtd_log = np.array(dtd_log)

    time_axis = np.arange(len(coherence_log))*block/ fs

    
    plt.figure(figsize=(12,6))

    plt.plot(time_axis,coherence_log,label="Avg Coherence",linewidth=2)
    plt.axhline(
        cfg.DTD_COHERENCE_THRESHOLD,
        color="r",
        linestyle="--",
        label="DTD Threshold"
    )

    plt.step(
        time_axis,
        dtd_log,
        where="post",
        label="Double-Talk Decision",
        alpha=0.6
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Value")
    plt.title("Coherence-Based Double-Talk Detection")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(results_dir,"dtd_behavior.png")
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(f"Saved DTD behavior plot to {save_path}")


if __name__ == "__main__":
    main()
