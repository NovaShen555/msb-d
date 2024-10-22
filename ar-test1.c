#include <arduinoFFT.h>

#define SAMPLES 1024 // 必须为2的幂
#define SAMPLING_FREQUENCY 44100  // 采样频率

arduinoFFT FFT = arduinoFFT();

double vReal[SAMPLES]; // 实部
double vImag[SAMPLES]; // 虚部（在Arduino FFT库中实际不需要）
double vReal2[SAMPLES]; // 第二个信号的实部

void calculate_delay(double* signal1, double* signal2, double sampling_frequency, int N, double* delay, double* R12_shift) {
    // 将信号1和信号2复制到vReal数组中
    for (int i = 0; i < N; i++) {
        vReal[i] = signal1[i];
        vReal2[i] = signal2[i];
    }

    // 对信号1执行FFT
    FFT.Windowing(vReal, N, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
    FFT.Compute(vReal, vImag, N, FFT_FORWARD);

    // 对信号2执行FFT
    FFT.Windowing(vReal2, N, FFT_WIN_TYP_HAMMING, FFT_FORWARD);
    FFT.Compute(vReal2, vImag, N, FFT_FORWARD);

    // 计算互功率谱（G = signal1_fft * conj(signal2_fft)）
    for (int i = 0; i < N; i++) {
        double G_real = vReal[i] * vReal2[i] + vImag[i] * vImag[i];
        double G_imag = vImag[i] * vReal2[i] - vReal[i] * vImag[i];
        
        // 加权（避免零除）
        double w = 1.0 / max(sqrt(G_real * G_real + G_imag * G_imag), 0.1);
        w = constrain(w, 0.1, 4.0);

        // 应用加权后逆FFT
        vReal[i] = G_real * w;
        vImag[i] = G_imag * w;
    }

    // 执行逆FFT，得到互相关函数
    FFT.Compute(vReal, vImag, N, FFT_REVERSE);
    
    // 计算时延
    int peak_index = 0;
    double max_value = 0;
    
    for (int i = 0; i < N; i++) {
        double magnitude = sqrt(vReal[i] * vReal[i] + vImag[i] * vImag[i]);
        if (magnitude > max_value) {
            max_value = magnitude;
            peak_index = i;
        }
    }

    // 零频平移
    int half_N = N / 2;
    int shifted_index = (peak_index - half_N) % N;

    // 计算时延
    *delay = (double)(-shifted_index) / sampling_frequency;

    // 返回互相关结果R12_shift
    for (int i = 0; i < N; i++) {
        R12_shift[i] = sqrt(vReal[i] * vReal[i] + vImag[i] * vImag[i]); // 互相关的幅度
    }
}

void setup() {
    Serial.begin(115200);

    // 示例信号，假设已经通过ADC采集了两组数据
    double signal1[SAMPLES];
    double signal2[SAMPLES];

    // 初始化信号
    for (int i = 0; i < SAMPLES; i++) {
        signal1[i] = sin(2 * PI * 440 * i / SAMPLING_FREQUENCY);
        signal2[i] = sin(2 * PI * 440 * i / SAMPLING_FREQUENCY + 0.1); // 加入一些相位差
    }

    double delay;
    double R12_shift[SAMPLES];

    // 计算时延和互相关
    calculate_delay(signal1, signal2, SAMPLING_FREQUENCY, SAMPLES, &delay, R12_shift);

    // 输出时延和互相关结果
    Serial.print("Time delay: ");
    Serial.println(delay, 8);
    
    // 输出互相关的最大值（可视化或调试用）
    for (int i = 0; i < SAMPLES; i++) {
        Serial.print(R12_shift[i], 8);
        Serial.print(", ");
    }
}

void loop() {
    // 主循环不需要做额外操作
}
