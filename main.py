import numpy as np
import config.aec_config as cfg
from core.overlap_save import OverlapSave
from core.pbfdaf import PBFDAF
from core.adaptive_update import pbfda_nlms_update
from dtd.coherence_dtd import CoherenceDTD
from nlp.smoothing import GainSmoother
from metrics.erle import compute_erle
from simulation.generate_signals import generate_test_signals
from simulation.generate_signals import pre_emphasis


def main():

    print("Generating Test Signals...")
    far_end,mic_signal,clean_near_end,echo_component = generate_test_signals(duration_sec=5.0,fs=cfg.SAMPLE_RATE)
    far_end = pre_emphasis(far_end,alpha=0.97)
    mic_signal = pre_emphasis(mic_signal, alpha=0.97)
    echo_component = pre_emphasis(echo_component,alpha=0.97)
    num_blocks = len(far_end)//cfg.BLOCK_SIZE
    print(f"Total Blocks to process: {num_blocks}")

    os_far_end = OverlapSave(cfg.BLOCK_SIZE,cfg.FFT_SIZE)
    os_mic = OverlapSave(cfg.BLOCK_SIZE,cfg.FFT_SIZE)
    pbfdaf = PBFDAF(cfg.NUM_PARTITIONS,cfg.FFT_SIZE)
    dtd = CoherenceDTD(cfg.DTD_COHERENCE_THRESHOLD,cfg.DTD_SMOOTHING_FACTOR)
    num_bins = (cfg.FFT_SIZE//2) + 1
    gain_smoother = GainSmoother(num_bins=num_bins,alpha=0.7)

    output_signal = np.zeros_like(mic_signal)
    dtd_log = []

    print("Starting Offline Processing...")

    for i in range(num_blocks):

        idx_start = i*cfg.BLOCK_SIZE
        idx_end = idx_start +cfg.BLOCK_SIZE

        x_block = far_end[idx_start:idx_end]
        d_block = mic_signal[idx_start:idx_end]
        x_time_buf = os_far_end.process(x_block)
        d_time_buf = os_mic.process(d_block)
        X_f = np.fft.rfft(x_time_buf)
        D_f = np.fft.rfft(d_time_buf)

        is_double_talk = dtd.detect(X_f,D_f)
        dtd_log.append(is_double_talk)
        pbfdaf.update_input_history(X_f)
        Y_hat_f = pbfdaf.estimate_echo()

        y_hat_time = np.fft.irfft(Y_hat_f,n=cfg.FFT_SIZE)
        valid_start = cfg.FFT_SIZE-cfg.BLOCK_SIZE
        y_hat_block = y_hat_time[valid_start:]

        e_full = d_time_buf - y_hat_time
        E_f = np.fft.rfft(e_full, n=cfg.FFT_SIZE)
        # mu0 = cfg.STEP_SIZE
        # tau = 50
        # mu = mu0 *(1 - np.exp(-i / tau))
        # error_power = np.mean(np.abs(E_f)**2)
        # echo_power = np.mean(np.abs(Y_hat_f)**2)
        # if error_power < 0.01 *echo_power:
        #     mu *= 0.5

        
        mu = cfg.STEP_SIZE

        if i > 80: 
            error_power = np.mean(np.abs(E_f)**2)
            if error_power < 1e-3:
                mu *= 0.3


        pbfda_nlms_update(
            pbfdaf.W,
            pbfdaf.X_history,
            E_f,
            mu,
            cfg.EPSILON,
            adapt=not is_double_talk,
            # leakage=1e-3
        )
        if i == 50:
            print("||W|| =", np.linalg.norm(pbfdaf.W))
            print("||X_history|| =", np.linalg.norm(pbfdaf.X_history))
            print("||E_f|| =", np.linalg.norm(E_f))

        epsilon = 1e-10
        mag_E = np.abs(E_f)
        mag_Y = np.abs(Y_hat_f)

        est_residual = cfg.NLP_AGGRESSIVENESS * mag_Y
        raw_gain = (mag_E -est_residual)/(mag_E + epsilon)
        raw_gain = np.clip(raw_gain,cfg.NLP_MIN_GAIN_LINEAR,1.0)

        smoothed_gain = gain_smoother.smooth(raw_gain)
        E_enhanced_f = E_f * smoothed_gain
        e_enhanced_time = np.fft.irfft(E_enhanced_f,n=cfg.FFT_SIZE)
        e_final_block = e_enhanced_time[valid_start:]

        output_signal[idx_start:idx_end] = e_final_block

    print("Processing Completed")
    total_samples = len(mic_signal)
    # st_mask = np.ones(total_samples,dtype=bool)

    # dt_start = int(0.4*total_samples)
    # dt_end = int(0.6*total_samples)

    # st_mask[dt_start:dt_end] = False

    # convergence_margin = int(0.5*cfg.SAMPLE_RATE)
    # st_mask[:convergence_margin] = False

    # echo_ref_st = echo_component[st_mask]
    # residual_st = output_signal[st_mask]

    # final_erle =compute_erle(echo_ref_st, residual_st)


    block_energy = []
    for i in range(num_blocks):
        idx_start = i*cfg.BLOCK_SIZE
        idx_end = idx_start + cfg.BLOCK_SIZE
        block_energy.append(np.mean(far_end[idx_start:idx_end]**2))

    block_energy = np.array(block_energy)

    energy_threshold = 0.1*np.max(block_energy)

    active_blocks = block_energy > energy_threshold


    conv_blocks = int(0.5*cfg.SAMPLE_RATE/cfg.BLOCK_SIZE)
    active_blocks[:conv_blocks] = False

    active_mask = np.repeat(active_blocks,cfg.BLOCK_SIZE)


    if len(active_mask) < len(far_end):
        pad_len = len(far_end) - len(active_mask)
        active_mask = np.concatenate((active_mask, np.zeros(pad_len, dtype=bool)))


    echo_ref_st = echo_component[active_mask]
    residual_st = output_signal[active_mask]

    final_erle = compute_erle(echo_ref_st, residual_st)


    print("Results Summary")
    print(f"Sample Rate : {cfg.SAMPLE_RATE} Hz")
    print(f"Block Size : {cfg.BLOCK_SIZE}")
    print(f"Double-Talk Blocks : {sum(dtd_log)}")
    print(f"Global ERLE (ST) : {final_erle:.2f} dB")


if __name__ == "__main__":
    main()
