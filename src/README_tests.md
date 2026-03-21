# 音频转录 API 测试

本目录包含用于测试 `APIAudioTranscriptions` 类的测试文件。

## 测试文件说明

- `test_api_audio_transcriptions.py` - 实际测试音频转录 API（需要加载 whisper 模型）
- `test_api_audio_transcriptions_mock.py` - 使用模拟方式测试音频转录 API（不需要加载模型）
- `run_tests.py` - 运行测试的脚本

## 如何运行测试

### 运行所有测试

```bash
python run_tests.py
```

或者

```bash
./run_tests.py
```

### 只运行模拟测试（推荐用于 CI/CD）

这将只运行不需要加载实际模型的测试，速度更快：

```bash
python run_tests.py --mock-only
```

### 只运行实际测试（需要加载模型）

这将运行需要加载实际 whisper 模型的测试：

```bash
python run_tests.py --real-only
```

## 测试数据

测试使用 `assets/test_audio.ogg` 作为测试音频文件。

## 注意事项

- 实际测试需要加载 whisper 模型，可能需要较长时间，并占用大量内存
- 模拟测试适合用于快速验证 API 功能和异常处理
- 在 CI/CD 环境中，建议使用 `--mock-only` 选项运行测试 