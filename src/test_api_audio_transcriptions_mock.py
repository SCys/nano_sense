import os
import unittest
import io
import sys
from unittest.mock import patch, MagicMock
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application
from datetime import datetime

# 模拟 globals 模块
# 注意：必须在导入 api_audio_transcriptions 前进行模拟
# 创建一个模拟对象
mock_whisper_worker = MagicMock()

# 使用模块级别的模拟替换 globals 模块
sys.modules['globals'] = MagicMock()
sys.modules['globals'].whisper_worker = mock_whisper_worker

# 现在导入要测试的模块
from api_audio_transcriptions import APIAudioTranscriptions

class TestAPIAudioTranscriptionsMock(AsyncHTTPTestCase):
    """使用模拟测试音频转录API"""
    
    @patch('api_audio_transcriptions.datetime')
    def setUp(self, mock_datetime):
        """设置测试环境，模拟 datetime"""
        # 设置模拟的 datetime.now() 返回一个固定的时间
        mock_now = MagicMock()
        mock_datetime.now.return_value = mock_now
        mock_now.__sub__.return_value = datetime.timedelta(seconds=1)  # 模拟时间差
        
        super().setUp()
    
    def get_app(self):
        """构建测试应用"""
        return Application([
            (r"/v1/audio/transcriptions", APIAudioTranscriptions),
        ])
    
    def test_transcription_json_format(self):
        """测试JSON格式的音频转录，使用模拟"""
        # 模拟 segments 和 info
        mock_segment = MagicMock()
        mock_segment.id = 0
        mock_segment.start = 0.0
        mock_segment.end = 2.5
        mock_segment.text = "这是测试文本"
        
        mock_info = MagicMock()
        mock_info.language = "zh"
        mock_info.language_probability = 0.98
        mock_info.duration = 2.5
        
        # 设置模拟返回值
        mock_whisper_worker.transcribe.return_value = ([mock_segment], mock_info)
        
        # 获取测试音频文件路径
        test_audio_path = os.path.join('assets', 'test_audio.ogg')
        
        # 确认测试文件存在
        self.assertTrue(os.path.exists(test_audio_path), "测试音频文件不存在")
        
        # 读取测试音频文件
        with open(test_audio_path, 'rb') as f:
            audio_data = f.read()
        
        # 发送请求
        response = self.fetch(
            '/v1/audio/transcriptions',
            method='POST',
            body_producer=lambda write: write(
                b'--boundary\r\n'
                b'Content-Disposition: form-data; name="audio"; filename="test_audio.ogg"\r\n'
                b'Content-Type: audio/ogg\r\n\r\n' + 
                audio_data + 
                b'\r\n--boundary--\r\n'
            ),
            headers={
                'Content-Type': 'multipart/form-data; boundary=boundary',
            }
        )
        
        # 检查响应状态码
        self.assertEqual(response.code, 200)
        
        # 验证模拟函数被调用
        mock_whisper_worker.transcribe.assert_called_once()
        
        # 检查响应内容
        response_json = response.json()
        self.assertIn('text', response_json)
        self.assertEqual(response_json['text'], "这是测试文本")
        
    def test_transcription_text_format(self):
        """测试TEXT格式的音频转录，使用模拟"""
        # 模拟 segments 和 info
        mock_segment = MagicMock()
        mock_segment.id = 0
        mock_segment.start = 0.0
        mock_segment.end = 2.5
        mock_segment.text = "这是测试文本"
        
        mock_info = MagicMock()
        mock_info.language = "zh"
        mock_info.language_probability = 0.98
        mock_info.duration = 2.5
        
        # 设置模拟返回值
        mock_whisper_worker.transcribe.return_value = ([mock_segment], mock_info)
        
        # 获取测试音频文件路径
        test_audio_path = os.path.join('assets', 'test_audio.ogg')
        
        # 读取测试音频文件
        with open(test_audio_path, 'rb') as f:
            audio_data = f.read()
        
        # 发送请求
        response = self.fetch(
            '/v1/audio/transcriptions?response_format=text',
            method='POST',
            body_producer=lambda write: write(
                b'--boundary\r\n'
                b'Content-Disposition: form-data; name="audio"; filename="test_audio.ogg"\r\n'
                b'Content-Type: audio/ogg\r\n\r\n' + 
                audio_data + 
                b'\r\n--boundary--\r\n'
            ),
            headers={
                'Content-Type': 'multipart/form-data; boundary=boundary',
            }
        )
        
        # 检查响应状态码
        self.assertEqual(response.code, 200)
        
        # 验证模拟函数被调用
        mock_whisper_worker.transcribe.assert_called_once()
        
        # 检查响应内容
        response_text = response.body.decode('utf-8')
        self.assertEqual(response_text, "这是测试文本")
        
    def test_transcription_with_segments(self):
        """测试带分段信息的音频转录，使用模拟"""
        # 模拟 segments 和 info
        mock_segment1 = MagicMock()
        mock_segment1.id = 0
        mock_segment1.start = 0.0
        mock_segment1.end = 1.5
        mock_segment1.text = "这是第一段"
        
        mock_segment2 = MagicMock()
        mock_segment2.id = 1
        mock_segment2.start = 1.5
        mock_segment2.end = 3.0
        mock_segment2.text = "这是第二段"
        
        mock_info = MagicMock()
        mock_info.language = "zh"
        mock_info.language_probability = 0.98
        mock_info.duration = 3.0
        
        # 设置模拟返回值
        mock_whisper_worker.transcribe.return_value = ([mock_segment1, mock_segment2], mock_info)
        
        # 获取测试音频文件路径
        test_audio_path = os.path.join('assets', 'test_audio.ogg')
        
        # 读取测试音频文件
        with open(test_audio_path, 'rb') as f:
            audio_data = f.read()
        
        # 发送请求
        response = self.fetch(
            '/v1/audio/transcriptions?timestamp_granularities=segment',
            method='POST',
            body_producer=lambda write: write(
                b'--boundary\r\n'
                b'Content-Disposition: form-data; name="audio"; filename="test_audio.ogg"\r\n'
                b'Content-Type: audio/ogg\r\n\r\n' + 
                audio_data + 
                b'\r\n--boundary--\r\n'
            ),
            headers={
                'Content-Type': 'multipart/form-data; boundary=boundary',
            }
        )
        
        # 检查响应状态码
        self.assertEqual(response.code, 200)
        
        # 验证模拟函数被调用
        mock_whisper_worker.transcribe.assert_called_once()
        
        # 检查响应内容
        response_json = response.json()
        self.assertIn('text', response_json)
        self.assertEqual(response_json['text'], "这是第一段 这是第二段")
        self.assertIn('segments', response_json)
        self.assertEqual(len(response_json['segments']), 2)
        
        # 检查segments内容
        self.assertEqual(response_json['segments'][0]['id'], 0)
        self.assertEqual(response_json['segments'][0]['start'], 0.0)
        self.assertEqual(response_json['segments'][0]['end'], 1.5)
        self.assertEqual(response_json['segments'][0]['text'], "这是第一段")
        
        self.assertEqual(response_json['segments'][1]['id'], 1)
        self.assertEqual(response_json['segments'][1]['start'], 1.5)
        self.assertEqual(response_json['segments'][1]['end'], 3.0)
        self.assertEqual(response_json['segments'][1]['text'], "这是第二段")
    
    def test_exception_handling(self):
        """测试异常处理，使用模拟"""
        # 设置模拟抛出异常
        mock_whisper_worker.transcribe.side_effect = Exception("测试异常")
        
        # 获取测试音频文件路径
        test_audio_path = os.path.join('assets', 'test_audio.ogg')
        
        # 读取测试音频文件
        with open(test_audio_path, 'rb') as f:
            audio_data = f.read()
        
        # 发送请求
        response = self.fetch(
            '/v1/audio/transcriptions',
            method='POST',
            body_producer=lambda write: write(
                b'--boundary\r\n'
                b'Content-Disposition: form-data; name="audio"; filename="test_audio.ogg"\r\n'
                b'Content-Type: audio/ogg\r\n\r\n' + 
                audio_data + 
                b'\r\n--boundary--\r\n'
            ),
            headers={
                'Content-Type': 'multipart/form-data; boundary=boundary',
            }
        )
        
        # 检查响应状态码
        self.assertEqual(response.code, 200)
        
        # 验证模拟函数被调用
        mock_whisper_worker.transcribe.assert_called_once()
        
        # 检查响应内容
        response_json = response.json()
        self.assertIn('text', response_json)
        self.assertEqual(response_json['text'], "ASR Failed:测试异常")

if __name__ == '__main__':
    unittest.main() 