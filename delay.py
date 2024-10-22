import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
from scipy.signal import wiener

def calculate_delay(x1, x2, fs, N):
    """
    计算两个信号的时延并返回互相关结果 R12
    
    参数:
    x1: 第一个信号
    x2: 第二个信号
    fs: 采样频率 (Hz)
    N: 采样点数
    
    返回:
    delay: 时延 (秒)
    R12_shift: 互相关结果
    """
    # 傅里叶变换至频域
    x1_fft = fft(x1, N)
    x2_fft = fft(x2, N)
    
    # 计算互功率谱
    G = x1_fft * np.conj(x2_fft)
    
    # 相位变换加权
    w = 1.0 / np.abs(G)
    w[np.isinf(w)] = 0  # 防止除以0的情况

    w = np.clip(w, 0.1, 4.0)
    
    # 加权互功率谱
    Gw = G * w
    # Gw = G
    
    # 逆傅里叶变换得到互相关函数
    R12 = ifft(Gw)
    
    # 零频平移
    R12_shift = fftshift(R12)
    
    # 找峰值
    idx = np.argmax(np.abs(R12_shift))
    print(f"Peak index: {idx}")
    
    # 计算时延（N为采样点，fs为采样频率）
    sIndex = np.arange(-N // 2, N // 2)
    delay = -sIndex[idx] / fs
    
    return delay, R12_shift

# 示例
fs = 44100  # 采样频率
N = 1024    # 采样点数
t = np.linspace(0, 1, fs)
signal1_orig = np.sin(2 * np.pi * 440 * t)[:N]  # 440Hz 正弦波信号
signal2_orig = np.sin(2 * np.pi * 440 * t + np.pi/5)[:N]  # 带相位差的信号2

# 添加随机白噪声
noise1 = np.random.normal(0, 0.5, signal1_orig.shape)  # 白噪声，均值为0，标准差为0.05
noise2 = np.random.normal(0, 0.5, signal2_orig.shape)  # 白噪声，均值为0，标准差为0.05

# 对信号添加噪声
signal1_noisy = signal1_orig + noise1
signal2_noisy = (signal2_orig + noise2) * 1.05  # 对 signal2 的振幅进行微小调整（增加1%

signal1_filtered = wiener(signal1_noisy)
signal2_filtered = wiener(signal2_noisy)

# 计算时延和互相关结果
delay, R12_shift = calculate_delay(signal1_filtered, signal2_filtered, fs, N)

# 绘图
fig, axs = plt.subplots(5, 1, figsize=(10, 12))

print(f"Time delay: {delay:.8f} seconds")
angle = delay * 2 * np.pi * 440
print(f"Phase difference: {angle:.4f} radians")

# # 绘制原始信号 signal1
# axs[0].plot(signal1_orig, label="Original Signal 1")
# axs[0].set_title("Original Signal 1")
# axs[0].set_ylabel("Amplitude")
# axs[0].legend()

# # 绘制原始信号 signal2
# axs[1].plot(signal2_orig, label="Original Signal 2")
# axs[1].set_title("Original Signal 2")
# axs[1].set_ylabel("Amplitude")
# axs[1].legend()

# 绘制带噪声信号 signal1
axs[0].plot(signal1_noisy, label="Noisy Signal 1")
axs[0].set_title("Noisy Signal 1 (with Gaussian Noise)")
axs[0].set_ylabel("Amplitude")
axs[0].legend()

# 绘制带噪声信号 signal2
axs[1].plot(signal2_noisy, label="Noisy Signal 2")
axs[1].set_title("Noisy Signal 2 (with Gaussian Noise)")
axs[1].set_ylabel("Amplitude")
axs[1].legend()

# 绘制经过维纳滤波的信号 signal1
axs[2].plot(signal1_filtered, label="Filtered Signal 1 (Wiener)")
axs[2].set_title("Filtered Signal 1 (Wiener)")
axs[2].set_ylabel("Amplitude")
axs[2].legend()

# 绘制经过维纳滤波的信号 signal2
axs[3].plot(signal2_filtered, label="Filtered Signal 2 (Wiener)")
axs[3].set_title("Filtered Signal 2 (Wiener)")
axs[3].set_ylabel("Amplitude")
axs[3].legend()

# 绘制互相关结果 R12
axs[4].plot(np.abs(R12_shift), label="Cross-correlation (R12)")
axs[4].set_title("Cross-correlation (R12)")
axs[4].set_ylabel("Magnitude")
axs[4].set_xlabel("Samples")
axs[4].legend()

plt.tight_layout()
plt.show()

