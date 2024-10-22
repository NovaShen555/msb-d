import pyaudio
import numpy as np
import matplotlib.pyplot as plt

# 定义一些音频参数
CHUNK = 1024  # 每个数据块的大小
FORMAT = pyaudio.paInt16  # 16位音频格式
CHANNELS = 1  # 单声道
RATE = 44100  # 采样率（每秒采样次数）
DURATION = 2  # 振幅图展示最近2秒的数据

# 计算2秒内的总样本数
TOTAL_SAMPLES = RATE * DURATION

# 实例化 PyAudio
p = pyaudio.PyAudio()

# 打开麦克风流
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# 初始化保存2秒内的音频数据的数组
data_buffer = np.zeros(TOTAL_SAMPLES, dtype=np.int16)

# 初始化实时振幅图窗口
plt.ion()
fig_amplitude, ax_amplitude = plt.subplots()
x_time = np.arange(0, TOTAL_SAMPLES)  # 时间轴数据
line_amplitude, = ax_amplitude.plot(x_time, np.zeros(TOTAL_SAMPLES), 'b')
ax_amplitude.set_ylim(-6000, 6000)  # 振幅范围设置为 [-6000, 6000]
ax_amplitude.set_xlim(0, TOTAL_SAMPLES)
ax_amplitude.set_title("Real-time Microphone Amplitude (Last 2 seconds)")
ax_amplitude.set_xlabel("Samples")
ax_amplitude.set_ylabel("Amplitude")

# 初始化实时频谱图窗口
fig_fft, ax_fft = plt.subplots()
x_fft = np.fft.rfftfreq(CHUNK, 1.0 / RATE)  # 计算频率坐标轴
line_fft, = ax_fft.plot(x_fft, np.zeros(len(x_fft)), 'r')
ax_fft.set_xlim(20, RATE / 2)  # 限制频率范围到 [20Hz, Nyquist 频率]
ax_fft.set_ylim(0, 1000)  # 根据你的音频范围调整
ax_fft.set_title("Real-time Audio Frequency Spectrum")
ax_fft.set_xlabel("Frequency (Hz)")
ax_fft.set_ylabel("Amplitude")

print("Recording...")

# 实时更新麦克风音频数据
try:
    while True:
        # 从麦克风读取数据
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # 将数据转换为 NumPy 数组
        new_samples = np.frombuffer(data, dtype=np.int16)
        
        # 1. 实时更新振幅图
        # 更新数据缓冲区，移除最早的数据并添加最新的数据
        data_buffer = np.roll(data_buffer, -CHUNK)
        data_buffer[-CHUNK:] = new_samples
        line_amplitude.set_ydata(data_buffer)
        
        # 2. 实时更新频谱图 (FFT)
        fft_data = np.fft.rfft(new_samples)
        fft_amplitude = np.abs(fft_data) / CHUNK  # 归一化并取绝对值
        line_fft.set_ydata(fft_amplitude)
        
        # 使用 plt.pause() 强制更新图像
        plt.pause(0.01)

except KeyboardInterrupt:
    print("Recording stopped")

finally:
    # 停止并关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()
    plt.ioff()  # 关闭交互模式
    plt.show()  # 确保绘图窗口关闭前显示最终图像
