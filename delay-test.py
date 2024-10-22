
from pydub import AudioSegment
from scipy.io.wavfile import write

def mp3_to_amplitude(mp3_file):
    # 将MP3文件加载为音频数据
    audio = AudioSegment.from_mp3(mp3_file)

    # 将音频数据转为原始音频数组
    samples = np.array(audio.get_array_of_samples())

    # 如果是立体声（双通道），则取平均值以获取单声道的振幅
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)

    # 返回音频数据和采样率
    return samples, audio.frame_rate


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


def save_to_wav(sample, fr, output):
    # 归一化处理，使数据适合WAV格式
    sample = np.int16(sample / np.max(np.abs(sample)) * 32767)

    # 保存为WAV文件
    write(output, fr, sample)
    print(f"WAV file saved as: {output}")

# 示例
# fs = 44100  # 采样频率
# N = 1024    # 采样点数
# t = np.linspace(0, 1, fs)
# signal1_orig = np.sin(2 * np.pi * 440 * t)[:N]  # 440Hz 正弦波信号
# signal2_orig = np.sin(2 * np.pi * 440 * t + np.pi/5)[:N]  # 带相位差的信号2

mp3_file = "soud\\1-1.mp3"
samples, frame_rate = mp3_to_amplitude(mp3_file)

# 添加随机相位 
signal1_orig = samples
signal2_orig = np.roll(signal1_orig, 100)
print(signal1_orig.shape)


# 添加随机白噪声
noise1 = np.random.normal(0, 500, signal1_orig.shape)  # 白噪声，均值为0，标准差为0.05
noise2 = np.random.normal(0, 500, signal2_orig.shape)  # 白噪声，均值为0，标准差为0.05

# 对信号添加噪声
signal1_noisy = signal1_orig + noise1
signal2_noisy = (signal2_orig + noise2) * 1.05  # 对 signal2 的振幅进行微小调整（增加1%

save_to_wav(signal1_noisy, frame_rate, "result\\signal1_noisy.wav")

signal1_filtered = wiener(signal1_noisy)
signal2_filtered = wiener(signal2_noisy)

save_to_wav(signal1_filtered, frame_rate, "result\\signal1_filtered.wav")

# 计算时延和互相关结果
delay, R12_shift = calculate_delay(signal1_filtered, signal2_filtered, frame_rate, 44160)

# 绘图
fig, axs = plt.subplots(5, 1, figsize=(10, 12))

print(f"Time delay: {delay:.8f} seconds")
angle = delay * 2 * np.pi * 440
print(f"Phase difference: {angle:.4f} radians")

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