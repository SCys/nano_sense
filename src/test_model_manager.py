"""ModelManager 自动下线与引用计数保护测试"""
import time
import unittest
from unittest.mock import MagicMock

from globals import ModelManager


class TestAutoUnload(unittest.TestCase):
    def _make_manager(self, timeout=2.0, name="test"):
        load_count = {"n": 0}

        def loader():
            load_count["n"] += 1
            return object()

        m = ModelManager(loader, timeout_seconds=timeout, name=name)
        return m, load_count

    def test_get_loads_once(self):
        """多次 get 只加载一次，并刷新访问时间"""
        m, count = self._make_manager()
        a = m.get()
        b = m.get()
        self.assertIs(a, b)
        self.assertEqual(count["n"], 1)

    def test_use_context_manager_tracks_active_count(self):
        """use() 上下文管理器应正确增加和减少活跃计数"""
        m, count = self._make_manager()
        self.assertEqual(m.active_count, 0)
        with m.use() as inst:
            self.assertIsNotNone(inst)
            self.assertEqual(m.active_count, 1)
            with m.use():
                self.assertEqual(m.active_count, 2)
            self.assertEqual(m.active_count, 1)
        self.assertEqual(m.active_count, 0)

    def test_unload_blocked_when_active(self):
        """模型处于活跃使用状态时，即使时间超过 timeout 也绝对不能被看门狗卸载"""
        m, count = self._make_manager(timeout=0.2)
        with m.use():
            time.sleep(0.4)  # 超过 timeout
            # 活跃状态下 unload_if_idle 应该返回 False 并跳过卸载
            self.assertFalse(m.unload_if_idle())
            self.assertTrue(m.loaded)
            self.assertEqual(count["n"], 1)

        # 退出使用状态后，等待超时再次调用 unload_if_idle 可以成功卸载
        time.sleep(0.3)
        self.assertTrue(m.unload_if_idle())
        self.assertFalse(m.loaded)

    def test_unload_if_idle_not_yet_timeout(self):
        """未超时不卸载"""
        m, count = self._make_manager(timeout=60.0)
        m.get()
        self.assertFalse(m.unload_if_idle())
        self.assertTrue(m.loaded)
        self.assertEqual(count["n"], 1)

    def test_unload_if_idle_after_timeout(self):
        """超过 timeout 后由 unload_if_idle 主动下线，下次 get 重新加载"""
        m, count = self._make_manager(timeout=0.5)
        m.get()
        self.assertTrue(m.loaded)
        time.sleep(0.7)  # 超过 timeout
        self.assertTrue(m.unload_if_idle())
        self.assertFalse(m.loaded)
        # 下次访问重新加载
        m.get()
        self.assertEqual(count["n"], 2)

    def test_unload_when_never_accessed_or_already_offline(self):
        """从未加载 / 已下线时不应重复卸载"""
        m, count = self._make_manager(timeout=0.1)
        self.assertFalse(m.unload_if_idle())  # 从未加载
        m.get()
        time.sleep(0.2)
        self.assertTrue(m.unload_if_idle())
        self.assertFalse(m.unload_if_idle())  # 已下线
        self.assertEqual(count["n"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
