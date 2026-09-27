# Submission check

This file is the final check against the assignment. The behaviour it describes is what the code does.

## Requirements

| # | Requirement | Where it is satisfied |
|---|---|---|
| 1 | Python 3.10+ | `pyproject.toml` requires `>=3.10` |
| 2 | LangChain tool calling | `llm.bind_tools` and the loop in `app/agent/agent.py` |
| 3 | An LLM provider | OpenAI through `app/agent/llm.py`, key in `.env` |
| 4 | The model chooses tools | No keyword `if` on the user message. Tests use a scripted model that returns tool calls. |
| 5 | At least two custom tools | `calculator`, `read_file`, and `read_skill` |
| 6 | CLI interaction | `python -m app.main` |
| 7 | Structured output | `AgentResponse` in `app/models/schemas.py`. The CLI prints the answer and the tools used. |
| 8 | Errors do not crash the process | Tool and API failures become a reply. Details go to `agent.log`. |
| 9 | Separate modules | `agent`, `tools`, `models`, `config`, `main`, `tests` |
| 10 | Tests and documentation | `pytest` without an API key. `README.md` and this file. |

## Five demonstration prompts

Restart the CLI before the presentation so it loads the current code. Type `exit` between unrelated demos if you want a fresh conversation. Stay in one session for demo 4, because it uses the previous answer.

1. **Calculator.** Input: `What is 125 / 5?` The model should call `calculator`. Expected result: `25`, and `Tools used: calculator`. Shows tool selection and a custom tool.

2. **File reader.** Input: `Read example.txt`. The model should call `read_file`. Expected result: a short summary of that file, and `Tools used: read_file`. Shows the second custom tool and the `data/` limit.

3. **No tool.** Input: `What is LangChain?` The model should answer with no tool line. Shows that tool use is a decision, not something that happens on every message.

4. **Two steps, then a follow-up.** Input: `Read numbers.json and add the values`. Expect `Reading numbers.json...`, then `Using calculator...`, then `150`. Then type: `Now divide that by 3`. Expect `50` from the calculator, using the previous answer. Shows the loop running more than once, and that the CLI keeps history.

5. **Skill, then an error that does not crash.** Input: `a=1, b=20, c=30. What are the real roots?` Expect `Using skill quadratic-equation...` and one or more calculator calls. Then type: `Read missing.txt`. Expect a reply that the file does not exist, and the `You:` prompt still waiting. Shows skills plus error handling.

If a live answer is worded differently, judge the tool lines, not the exact sentence.

## Two-minute explanation

This is a developer assistant. I type a request in the terminal. `app/main.py` sends that text to `run_agent`.

`run_agent` does not look for words like "calculate" or "read". It gives the OpenAI model three tools: a calculator, a file reader locked to the `data` folder, and a skill reader locked to the `skills` folder. Skills are markdown instructions. The model sees only their names and descriptions, and it opens one when the request matches.

The loop is: ask the model, run the tool it chose, give the result back, and ask again. When the model returns normal text, that is the answer. After six rounds it stops, so a bad tool choice cannot run forever. Failures come back as text, so a missing file or a division by zero does not kill the program.

The result is an `AgentResponse`: the answer, the tools that ran, and whether it succeeded. The CLI prints the answer and a line such as `Tools used: read_skill, calculator X2`. Tests use a fake model, so `pytest` checks this loop without an API key.
