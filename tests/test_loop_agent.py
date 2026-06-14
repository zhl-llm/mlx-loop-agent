import unittest

from mlx_loop_agent import LangChainLoopAgent


class CountingRunnable:
    def __init__(self, stop_at: int) -> None:
        self._stop_at = stop_at
        self._calls = 0

    def invoke(self, input_state):
        self._calls += 1
        value = input_state.get("value", 0) + 1
        return {
            "state": {"value": value},
            "done": self._calls >= self._stop_at,
        }


class BadRunnable:
    def invoke(self, input_state):
        return "not-a-dict"


class TestLangChainLoopAgent(unittest.TestCase):
    def test_stops_when_done_is_true(self):
        agent = LangChainLoopAgent(CountingRunnable(stop_at=3), max_iterations=10)

        result = agent.run({"value": 0})

        self.assertTrue(result.finished)
        self.assertEqual(result.iterations, 3)
        self.assertEqual(result.state["value"], 3)
        self.assertEqual(len(result.steps), 3)

    def test_stops_at_max_iterations_when_not_done(self):
        agent = LangChainLoopAgent(CountingRunnable(stop_at=50), max_iterations=4)

        result = agent.run({"value": 0})

        self.assertFalse(result.finished)
        self.assertEqual(result.iterations, 4)
        self.assertEqual(result.state["value"], 4)
        self.assertEqual(len(result.steps), 4)

    def test_raises_for_non_mapping_response(self):
        agent = LangChainLoopAgent(BadRunnable(), max_iterations=1)

        with self.assertRaises(TypeError):
            agent.run()

    def test_rejects_invalid_max_iterations(self):
        with self.assertRaises(ValueError):
            LangChainLoopAgent(CountingRunnable(stop_at=1), max_iterations=0)


if __name__ == "__main__":
    unittest.main()
