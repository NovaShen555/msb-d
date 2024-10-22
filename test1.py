import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import sys
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


def plot_amplitude_over_time(samples, frame_rate, start_time=0, duration=None, dpi=400):
    # 确定音频的持续时间
    total_duration = len(samples) / frame_rate

    # 如果没有指定绘图的持续时间，则绘制整个音频文件
    if duration is None or duration > total_duration:
        duration = total_duration

    # 计算绘图的起止样本点
    start_sample = int(start_time * frame_rate)
    end_sample = int((start_time + duration) * frame_rate)

    # 确定绘制的样本数据
    plot_samples = samples[start_sample:end_sample]

    # 创建时间轴
    time_axis = np.linspace(start_time, start_time + duration, num=len(plot_samples))

    # 增加绘图的DPI以提高分辨率
    plt.figure(figsize=(10, 6), dpi=dpi)
    plt.plot(time_axis, plot_samples, color='b', linewidth=0.5)
    plt.title(f"Amplitude Over Time ({start_time}s to {start_time + duration}s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


def save_to_wav(sample, fr, output):
    # 归一化处理，使数据适合WAV格式
    sample = np.int16(sample / np.max(np.abs(sample)) * 32767)

    # 保存为WAV文件
    write(output, fr, sample)
    print(f"WAV file saved as: {output}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <mp3_file> <output>")
        sys.exit(1)

    mp3_file = sys.argv[1]
    output_wav_file = sys.argv[2]
    samples, frame_rate = mp3_to_amplitude(mp3_file)

    plot_amplitude_over_time(samples, frame_rate)

    # 将振幅数组保存为WAV文件
    # save_to_wav(samples, frame_rate, output_wav_file)
