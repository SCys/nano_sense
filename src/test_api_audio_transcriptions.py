import os
import unittest
import io
from tornado.testing import AsyncHTTPTestCase
from tornado.web import Application

from api_audio_transcriptions import APIAudioTranscriptions
from globals import whisper_worker

class TestAPIAudioTranscriptions(AsyncHTTPTestCase):
    """测试音频转录API"""
    
    def get_app(self):
        """构建测试应用"""
        return Application([
            (r"/v1/audio/transcriptions", APIAudioTranscriptions),
        ])
    
    def test_transcription_json_format(self):
        """测试JSON格式的音频转录"""
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
        
        # 检查响应内容
        response_json = response.json()
        self.assertIn('text', response_json)
        self.assertIsInstance(response_json['text'], str)
        self.assertGreater(len(response_json['text']), 0)
        
    def test_transcription_text_format(self):
        """测试TEXT格式的音频转录"""
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
        
        # 检查响应内容
        response_text = response.body.decode('utf-8')
        self.assertIsInstance(response_text, str)
        self.assertGreater(len(response_text), 0)
        
    def test_transcription_with_segments(self):
        """测试带分段信息的音频转录"""
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
        
        # 检查响应内容
        response_json = response.json()
        self.assertIn('text', response_json)
        self.assertIn('segments', response_json)
        self.assertIsInstance(response_json['segments'], list)
        self.assertGreater(len(response_json['segments']), 0)
        
        # 检查segments内容
        segment = response_json['segments'][0]
        self.assertIn('id', segment)
        self.assertIn('start', segment)
        self.assertIn('end', segment)
        self.assertIn('text', segment)
    
    def test_empty_request(self):
        """测试空请求"""
        # 发送请求
        response = self.fetch(
            '/v1/audio/transcriptions',
            method='POST',
            body='',
            headers={
                'Content-Type': 'multipart/form-data; boundary=boundary',
            }
        )
        
        # 检查响应状态码
        self.assertEqual(response.code, 200)
        
        # 检查响应内容
        response_json = response.json()
        self.assertIn('text', response_json)
        self.assertEqual(response_json['text'], '')

if __name__ == '__main__':
    unittest.main() 