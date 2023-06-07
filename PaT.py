import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

class PanTompkinsDetector:

    def __init__(self, fs):

        self.fs = fs
        self.plt_period = [10, 20]

    def detect_qrs(self, signal):
        
        normalized_signal, smoothed_signal = self.pre_process(signal)

        differentiated_signal = np.diff(smoothed_signal)

        squared_signal = differentiated_signal ** 2
        # elf.plt_signal(squared_signal, tag='Sqaured')
        
        integrated_signal = self.moving_integration(squared_signal, window_length=0.15) # 移动窗口积分
        # self.plt_signal(integrated_signal, tag='Integrated')
        
        qrs_peaks = self.find_qrs_peaks(integrated_signal) # 寻找QRS波峰
        # self.plt_qrs(smoothed_signal, qrs_peaks, tag='Signal', ref=normalized_signal)

        self.plt_all(smoothed_signal, tag='Signal', qrs=qrs_peaks, ref=normalized_signal, n_sub=4, font_size=5)
        plt.show()

        return qrs_peaks

    def pre_process(self, signal, N=6):
        
        normalized_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))   # 预处理，将信号标准化到0-1之间
        
        filtered_signal = self.signal_filter(normalized_signal, type='highpass', cutoff=5, N=N)   # 高通滤波，去除基线漂移
        
        smoothed_signal = self.signal_filter(filtered_signal, type='lowpass', cutoff=15, N=N)   # 低通滤波，平滑信号

        # smoothed_signal = self.signal_filter(normalized_signal, type='band', cutoff=[5, 15], N=6)   # 带通

        return normalized_signal, smoothed_signal

    def signal_filter(self, signal, type, cutoff, N=6):
        
        nyquist = 0.5 * self.fs
        normalized_cutoff = np.array(cutoff) / nyquist
        b, a = sig.butter(N, normalized_cutoff, btype=type, analog=False)

        filtered_signal = sig.filtfilt(b, a, signal)

        return filtered_signal

    def moving_integration(self, signal, window_length=0.15):

        window_size = int(window_length * self.fs * 0.5) * 2 + 1   # 确保窗口长度为奇数
        window = np.ones(window_size) / window_size
        
        integrated_signal = np.convolve(signal, window, mode='same')   # 使用卷积进行移动窗口积分

        return integrated_signal

    def find_qrs_peaks(self, signal, threshold=0.33, rest_time=0.2):
        
        peak_threshold = threshold * np.max(signal) # 计算QRS波峰的阈值

        qrs_peaks = []
        
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > peak_threshold:
                qrs_peaks.append(i)

        corrected_peaks = []   # 进行R波修正，保留峰值最高的一个点
        rest_range = int(rest_time * self.fs)
        for peak in qrs_peaks:
            if signal[peak] == np.max(signal[max(peak-rest_range, 0):min(peak+rest_range, len(signal)-1)]):
                corrected_peaks.append(peak)
        
        return corrected_peaks

    def calc_heart_rate(self, signal, qrs_peaks, n_window=2):

    	rate = len(qrs_peaks) * self.fs / len(signal)

    	peak_cnt = 0
    	heart_rate = []

    	for t in range(len(signal)):

    		heart_rate.append(rate)

    		if peak_cnt == len(qrs_peaks) - n_window - 1:
    			continue
    		if qrs_peaks[peak_cnt] == t:
    			peak_cnt += 1
    		if peak_cnt < n_window:
    			continue
    		rate = n_window * 2 * 60 * self.fs / (qrs_peaks[peak_cnt + n_window] - qrs_peaks[peak_cnt - n_window])

    	return heart_rate

    def plt_signal(self, signal, tag='Signal'):

        st, ed = np.array(self.plt_period) * self.fs
        t = np.array(list(range(st, ed))) / self.fs

        plt.figure()
        plt.plot(t, signal[st:ed], label=tag)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()

    def plt_qrs(self, signal, qrs_peaks, tag='Signal', ref=[]):

        st, ed = np.array(self.plt_period) * self.fs
        t = np.array(list(range(st, ed))) / self.fs
        t_qrs, s_qrs = [], []

        for peak in qrs_peaks:
            if peak < st:
                continue
            if peak >= ed:
                break
            t_qrs.append(peak / self.fs)
            s_qrs.append(signal[peak])

        plt.figure()
        plt.plot(t, signal[st:ed], label=tag)
        if len(ref) > 0:
            plt.plot(t, ref[st:ed], alpha=0.3, label='Ref')
        plt.plot(t_qrs, s_qrs, 'ro', label='QRS Peaks')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()

    def plt_all(self, signal, tag='Signal', qrs=[], ref=[], n_sub=4, font_size=5):

        p_sub = int(np.floor(len(signal) / n_sub))
        heart_rate = self.calc_heart_rate(signal, qrs)
        print(heart_rate)
        normalized_rate = (np.array(heart_rate) - 75) / 75

        for i in range(n_sub):

            st = p_sub * i
            ed = p_sub * (i + 1)
            t = np.array(list(range(st, ed))) / self.fs
            plt.subplot(n_sub, 1, i + 1)

            plt.plot(t, signal[st:ed], label=tag)
            plt.plot(t, normalized_rate[st:ed], label='HeartRate')


            if len(qrs) > 0:
                t_qrs, s_qrs = [], []
                for peak in qrs:
                    if peak < st:
                        continue
                    if peak >= ed:
                        break
                    t_qrs.append(peak / self.fs)
                    s_qrs.append(signal[peak])
                plt.plot(t_qrs, s_qrs, 'ro', marker='o', markerfacecolor='none', label='QRS Peaks')

            if len(ref) > 0:
                plt.plot(t, ref[st:ed], alpha=0.5, label='Ref')

            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.xlabel('Time (s)', fontsize=font_size)
            plt.ylabel('Amplitude', fontsize=font_size)

