#!/usr/bin/env python3
"""
音频转录API测试运行脚本
"""
import unittest
import argparse
import sys

def run_all_tests():
    """运行所有测试"""
    # 发现并运行所有测试
    test_suite = unittest.defaultTestLoader.discover('.', pattern='test_*.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    return result.wasSuccessful()

def run_mock_tests_only():
    """只运行模拟测试"""
    # 发现并运行只包含模拟的测试
    test_suite = unittest.defaultTestLoader.discover('.', pattern='test_*_mock.py')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    return result.wasSuccessful()

def run_real_tests_only():
    """只运行实际测试（非模拟）"""
    # 加载特定的测试文件
    test_suite = unittest.defaultTestLoader.loadTestsFromName('test_api_audio_transcriptions')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    return result.wasSuccessful()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行音频转录API测试')
    parser.add_argument('--mock-only', action='store_true', help='只运行模拟测试')
    parser.add_argument('--real-only', action='store_true', help='只运行实际测试（非模拟）')
    
    args = parser.parse_args()
    
    if args.mock_only and args.real_only:
        print("错误：不能同时指定 --mock-only 和 --real-only")
        sys.exit(1)
    
    if args.mock_only:
        success = run_mock_tests_only()
    elif args.real_only:
        success = run_real_tests_only()
    else:
        success = run_all_tests()
    
    # 根据测试结果设置退出码
    sys.exit(0 if success else 1) 