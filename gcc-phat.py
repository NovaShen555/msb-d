import numpy as np

def gcc_phat_weighted(sig1, sig2, fs, weighting='magnitude'):
    """
    使用加权的 GCC-PHAT 计算两个信号的时差（TDOA）
    
    参数:
    sig1: 第一个麦克风的音频信号
    sig2: 第二个麦克风的音频信号
    fs: 采样率 (Hz)
    weighting: 加权方式，'magnitude' 使用幅度加权
    
    返回:
    时差 (秒)
    """
    # 将信号转为频域
    n = len(sig1) + len(sig2)
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    
    # 计算互功率谱
    R = SIG1 * np.conj(SIG2)
    
    # 选择加权方式
    if weighting == 'magnitude':
        # 使用互功率谱的幅度进行加权
        W = np.abs(R)
        W[W == 0] = 1e-10  # 避免除以零
        R = R / W  # 加权处理
    
    elif weighting == 'none':
        # 不进行加权处理
        pass
    
    # 转回时域
    cross_corr = np.fft.irfft(R, n=n)
    
    # 找到最大相关性对应的时差
    max_shift = int(n / 2)
    cross_corr = np.concatenate((cross_corr[-max_shift:], cross_corr[:max_shift]))
    shift = np.argmax(np.abs(cross_corr)) - max_shift
    
    # 计算时差
    time_delay = shift / fs
    return time_delay

# 示例音频信号
fs = 44100  # 假设采样率为 44100Hz
signal1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, fs))  # 440Hz 正弦波信号
signal2 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, fs) + np.pi/200)  # 相位差90度

# 添加随机白噪声
noise1 = np.random.normal(0, 0.05, signal1.shape)  # 白噪声，均值为0，标准差为0.05
noise2 = np.random.normal(0, 0.05, signal2.shape)  # 白噪声，均值为0，标准差为0.05

# 对信号添加噪声
signal1_noisy = signal1 + noise1
signal2_noisy = (signal2 + noise2) * 1.05  # 对 signal2 的振幅进行微小调整（增加1%）


# 计算加权的时差
time_delay_weighted = gcc_phat_weighted(signal1, signal2, fs, weighting='none')
print(f"time delay: {time_delay_weighted} s")

# # 可进一步计算相位差
# f_max = 440  # 假设信号的主频为 440Hz
# phase_difference = 2 * np.pi * f_max * time_delay_weighted
# print(f"相位差: {phase_difference} 弧度")
