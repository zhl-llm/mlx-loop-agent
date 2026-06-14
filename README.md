# mlx-loop-agent

A minimal loop agent implemented in the LangChain style.

## Usage

```python
from mlx_loop_agent import LangChainLoopAgent

class MyRunnable:
    def invoke(self, input):
        step = input.get("step", 0) + 1
        return {
            "state": {"step": step},
            "done": step >= 3,
        }

agent = LangChainLoopAgent(MyRunnable(), max_iterations=10)
result = agent.run({"step": 0})

print(result.finished)    # True
print(result.iterations)  # 3
print(result.state)       # {'step': 3}
```

The runnable is expected to expose `invoke(state) -> dict` and may return:

- `state`: partial state updates merged into the running state
- `done`: boolean stop signal for the loop
