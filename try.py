import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # 或 "MacOSX"
import matplotlib.pyplot as plt


def fractal_noise_2d(shape=(512, 512), beta=1.8, seed=0):
    """
    频域 1/f^beta 分形噪声（近似 fBm / pink-ish noise）
    beta 越大 => 越“平滑/大块结构”，越小 => 越“颗粒/高频多”
    """
    rng = np.random.default_rng(seed)
    h, w = shape

    # 复高斯白噪声（频域）
    real = rng.standard_normal((h, w))
    imag = rng.standard_normal((h, w))
    z = real + 1j * imag

    # 频率网格（以 cycles/pixel 表示）
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx * fx + fy * fy)

    # 1/f^beta 幅度谱（避免 f=0 发散）
    f[0, 0] = np.inf
    amp = 1.0 / (f ** (beta / 2.0))  # /2：把功率谱beta映射到幅度

    # 应用滤波并反变换回空间域
    z_filt = z * amp
    x = np.fft.ifft2(z_filt).real

    # 归一化到 [0, 1]
    x -= x.min()
    x /= (x.max() + 1e-12)
    return x

# ===== demo =====
img = fractal_noise_2d((512, 512), beta=3.8, seed=42)

plt.figure(figsize=(6, 6))
plt.imshow(img, interpolation="nearest")
plt.axis("off")
plt.title("Fractal Noise (1/f^beta)")
plt.show()