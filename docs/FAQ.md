# PersonaLive 安装使用 FAQ

---

## 问题 1: av 包安装失败 - 缺少 FFmpeg 库

### 错误信息

```bash
(personalive) root@AI-31:~/PersonaLive# pip install -r requirements_base.txt
Collecting torch==2.1.0 (from -r requirements_base.txt (line 1))
  Using cached torch-2.1.0-cp310-cp310-manylinux1_x86_64.whl.metadata (25 kB)
Collecting torchvision==0.16.0 (from -r requirements_base.txt (line 2))
  Using cached torchvision-0.16.0-cp310-cp310-manylinux1_x86_64.whl.metadata (6.6 kB)
Collecting accelerate==1.12.0 (from -r requirements_base.txt (line 3))
  Using cached accelerate-1.12.0-py3-none-any.whl.metadata (19 kB)
Collecting av==11.0.0 (from -r requirements_base.txt (line 4))
  Using cached av-11.0.0.tar.gz (3.7 MB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... error
  error: subprocess-exited-with-error

  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> [11 lines of output]
      Package libavformat was not found in the pkg-config search path.
      Perhaps you should add the directory containing `libavformat.pc'
      to the PKG_CONFIG_PATH environment variable
      Package 'libavformat', required by 'virtual:world', not found
      Package 'libavcodec', required by 'virtual:world', not found
      Package 'libavdevice', required by 'virtual:world', not found
      Package 'libavutil', required by 'virtual:world', not found
      Package 'libavfilter', required by 'virtual:world', not found
      Package 'swscale', required by 'virtual:world', not found
      Package 'swresample', required by 'virtual:world', not found
      pkg-config could not find libraries ['avformat', 'avcodec', 'avdevice', 'avutil', 'avfilter', 'swscale', 'swresample']
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed to build 'av' when getting requirements to build wheel
```

### 解决方案

安装 FFmpeg 开发库：

```bash
apt update
apt install -y libavformat-dev libavcodec-dev libavdevice-dev \
               libavutil-dev libavfilter-dev libswscale-dev libswresample-dev
```

---

## 问题 2: PyCUDA 安装失败 - 找不到 libcuda.so

### 错误信息

你遇到的是 PyCUDA 在编译阶段 **找不到 CUDA 驱动库 `libcuda.so`** 导致的链接失败。

```
/usr/bin/ld: cannot find -lcuda: No such file or directory
```

### 原因分析

`libcuda.so` 属于 **NVIDIA 驱动层面** 的库，而不是 CUDA Toolkit 里的 `libcudart.so`。

- 如果你只在系统里装了 `cuda-toolkit`，但 **没有安装 NVIDIA 驱动**（或驱动安装不完整），就会缺它。
- 如果你是在 **容器/无显卡环境** 里编译，也会缺它。

### 解决方案（按推荐顺序）

| 场景 | 操作 |
| :--- | :--- |
| **实体机/宿主机** 有 NVIDIA 显卡 | 1. 安装/更新 **官方驱动**（版本 ≥ 450 即可，建议最新）。Ubuntu 可用 `sudo apt install nvidia-driver-535` 或运行 `ubuntu-drivers autoinstall` 后重启。<br>2. 确认 `/usr/lib/x86_64-linux-gnu/` 或 `/usr/lib64/` 里出现 `libcuda.so*`（通常由驱动包提供）。 |
| **容器/云服务器** 无显卡 | 1. 如果 **不需要运行** CUDA 程序，只想把 PyCUDA 装完，可 **临时欺骗** 链接器： `sudo ln -s /dev/null /usr/lib/x86_64-linux-gnu/libcuda.so`（空链接让编译通过，但运行时会报错）。<br>2. 更干净的做法：用 `export CUDA_HOME=/usr/local/cuda` 并 `export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH`，然后 `pip install --no-cache-dir pycuda` 再试。 |
| **conda 环境** | 先 `conda install -c conda-forge cudatoolkit-dev`，再 `pip install pycuda`，conda 会把驱动 stub 放进环境，常能绕过链接问题。 |

### 验证驱动库

驱动装好后执行：

```bash
ls /usr/lib/x86_64-linux-gnu/libcuda.so*   # Ubuntu/Debian
# 或
ls /usr/lib64/libcuda.so*                   # CentOS/RHEL
```

有输出即代表驱动库已就位，再重新 `pip install pycuda` 即可顺利编译通过。

---

## 问题 2 补充: 驱动已安装但仍报错

### 现象

```bash
nvidia-driver-570 is already the newest version (570.195.03-0ubuntu1).
0 upgraded, 0 newly installed, 0 to remove and 4 not upgraded.

(personalive) root@AI-31:~/PersonaLive# ls /usr/lib/x86_64-linux-gnu/libcuda.so*
/usr/lib/x86_64-linux-gnu/libcuda.so.1
/usr/lib/x86_64-linux-gnu/libcuda.so.570.195.03
```

### 原因

驱动已经装好，但 PyCUDA 在链接阶段仍然报 `cannot find -lcuda`，根本原因是 **缺少"开发链接名"**：

```
/usr/lib/x86_64-linux-gnu/libcuda.so      ← 这个文件不存在
```

驱动包只提供了带版本号的 `.so.570.195.03` 和 `.so.1`，而 `ld` 默认找的是不带任何后缀的 `libcuda.so`。

### 解决方案

只需手动补一个软链接即可让编译通过：

```bash
sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.570.195.03 \
           /usr/lib/x86_64-linux-gnu/libcuda.so
```

再执行：

```bash
pip install --force-reinstall --no-cache-dir pycuda
```

应该就能顺利完成 `_driver` 模块的链接，不再出现 "cannot find -lcuda" 错误。
