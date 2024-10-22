import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import wiener


# 定义Butterworth低通滤波器
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# 参数设置
fs = 44100  # 采样率
cutoff = 10000  # 截止频率，假设我们希望滤除1kHz以上的高频噪声

# 添加高斯白噪声后的信号
t = np.linspace(0, 1, fs)
signal1 = np.sin(2 * np.pi * 440 * t)[:fs]  # 440Hz 正弦波信号
signal1 = np.sin(2 * np.pi * 440 * t)[:1024]  # 440Hz 正弦波信号
noise1 = np.random.normal(0, 0.2, signal1.shape)  # 高斯白噪声
signal1_noisy = signal1 + noise1

# 使用Butterworth低通滤波器消除噪声
signal1_filtered = butter_lowpass_filter(signal1_noisy, cutoff, fs)

# signal1_wiener_filtered = wiener(signal1_filtered)

# 绘图展示
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(signal1, label="Original Signal")
plt.title("Original Signal")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(signal1_noisy, label="Noisy Signal")
plt.title("Noisy Signal with Gaussian Noise")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(signal1_filtered, label="Filtered Signal")
plt.title("Filtered Signal (Butterworth Lowpass)")
plt.legend()

plt.tight_layout()
plt.show()
