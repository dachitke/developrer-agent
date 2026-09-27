# Developer Assistant AI Agent

A command-line agent that answers natural-language requests. The language model decides whether a tool is needed, runs that tool, and then writes the reply. The program does not choose tools by searching the user's sentence for keywords.

## What an AI agent is

A normal chatbot only produces text. An agent can also take actions. It looks at the request, chooses a tool when one would help, reads the tool result, and either calls another tool or answers. That cycle is the reasoning loop.

## What this agent does

This agent is a small developer assistant. It can:

- calculate an arithmetic expression
- read a `.txt`, `.md`, or `.json` file from the project's `data` folder
- open a skill from the `skills` folder and follow its instructions
- answer a general question without using a tool
- keep the conversation going, so a follow-up such as "now divide that by 3" still has the previous result
- stop after 6 rounds of tool calls instead of looping forever

After each answer that used a tool, the CLI prints the tools once each. A repeated tool is shown with a count, for example `calculator X3`.

## Architecture

```
You type a request
        |
        v
   app/main.py          CLI loop, progress lines, "Tools used"
        |
        v
   app/agent/agent.py   tool-calling loop (max 6 rounds)
        |
        +---- app/agent/llm.py -------- ChatOpenAI, built from .env
        |
        +---- tools
        |       calculator     safe arithmetic
        |       read_file      files inside data/ only
        |       read_skill     one SKILL.md from skills/
        |
        v
   AgentResponse        answer, tools used, success, error
```

The loop in `app/agent/agent.py` is:

```
user message
    -> LLM, with the three tools bound
    -> if the model requests a tool: run it, append the result, ask the LLM again
    -> if the model returns text: that text is the answer
    -> if this repeats 6 times: stop and tell the user
```

Progress lines such as `Using calculator...` or `Reading example.txt...` are printed while a tool runs. The model's private reasoning is not printed.

`AgentResponse` in `app/models/schemas.py` is the application's own structured result. The model is not asked to print JSON.

## Technologies

- Python 3.10+
- LangChain (`langchain-core` tool calling, `langchain-openai`)
- OpenAI chat models, default `gpt-4o-mini`
- Pydantic for settings, tool arguments, and `AgentResponse`
- python-dotenv for `.env`
- pytest, with a fake chat model so tests do not call the API

## Installation

From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Environment variables

```powershell
copy .env.example .env
```

Edit `.env` and set a real key. Do not commit `.env`.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | OpenAI key. The placeholder `your_api_key_here` is rejected. |
| `MODEL_NAME` | no | `gpt-4o-mini` | Chat model. It must support tool calling. |
| `TEMPERATURE` | no | `0` | Sampling temperature, from 0 to 2. |
| `DATA_DIR` | no | `data` | The only directory `read_file` may open. |
| `LOG_LEVEL` | no | `INFO` | Level written to `agent.log`. |

A missing key stops the program immediately with a clear message. The key is stored so that printing the settings does not reveal it. Technical errors, including stack traces, go to `agent.log`. The CLI shows a short message instead.

API calls are billed by OpenAI. `gpt-4o-mini` is the cheap default. Prepaid credit with auto-reload turned off is a way to cap spending.

## How to run

```powershell
python -m app.main
```

Type a request at `You:`. Type `exit` or `quit`, or press `Ctrl+C`, to leave. A failed request does not close the program.

## Example interactions

```
You: What is 125 / 5?
  Using calculator...
Agent: 125 / 5 is 25.
Tools used: calculator

You: Read numbers.json and add the values
  Reading numbers.json...
  Using calculator...
Agent: The values add up to 150.
Tools used: read_file, calculator

You: a=1, b=20, c=30. What are the quadratic roots?
  Using skill quadratic-equation...
  Using calculator...
Agent: ...
Tools used: read_skill, calculator X2

You: What is LangChain?
Agent: ...

You: Read missing.txt
  Reading missing.txt...
Agent: I couldn't read that file because it does not exist.
Tools used: read_file
```

The exact wording of an answer comes from the model. The tool lines come from the program.

## Available tools

| Tool | What it accepts | What it refuses |
|---|---|---|
| `calculator` | Numbers and `+ - * / // % **`, with parentheses | `eval`, names, function calls, huge exponents, division by zero |
| `read_file` | A file name inside `data/`, extension `.txt`, `.md`, or `.json` | `../`, absolute paths, other extensions, symlinks that leave `data/` |
| `read_skill` | A skill folder name, such as `quadratic-equation` | Paths that leave `skills/` |

Sample data files:

- `data/example.txt`
- `data/notes.md`
- `data/numbers.json` — values `10, 20, 30, 40, 50`

Skills are folders under `skills/`. Each folder has one `SKILL.md` file with a name, a description, and instructions. At startup the agent lists those descriptions. When a skill matches, the model calls `read_skill` and follows the file. Adding a skill is a new folder. It does not require a new Python tool.

Included skills:

- `quadratic-equation` — take `a`, `b`, and `c` from the user and use the calculator
- `project-summary` — read `example.txt` and `notes.md` before describing the project

## Error handling

Tool failures become text the model can explain. Examples: division by zero, a missing file, a bad argument, an unknown tool name, or an unexpected exception inside a tool. A network or API failure becomes a short message about the connection and the key. The Python process keeps running. Details are logged in `agent.log`, not printed as a traceback.

## Testing

```powershell
python -m pytest -q
```

Tests do not need an API key and do not call OpenAI. The agent tests use a scripted fake model, so they check the loop without spending credit. Covered areas include the calculator, the file reader, skills, tool selection, a missing file, a model failure, the step limit, and the CLI.

## Project structure

```
app/
  main.py                 CLI
  config.py               environment variables and logging
  agent/agent.py          tool-calling loop
  agent/llm.py            OpenAI client setup
  tools/calculator.py     calculator tool
  tools/file_reader.py    data-folder reader
  tools/skills.py         skills-folder reader
  models/schemas.py       AgentResponse
skills/                   SKILL.md instructions
data/                     files the agent is allowed to read
tests/                    pytest suite
.env.example              variable names only, no real key
requirements.txt          pinned packages
```

## Limitations

- The model chooses the tools. It can choose badly, for example by opening the same file again and again. The 6-step limit then stops it.
- It cannot browse the web, run shell commands, or read files outside `data/` and `skills/`.
- It only talks to OpenAI. There is no second provider.
- Answers are not deterministic even at temperature 0, so the words in a demo can vary.
- The CLI must be restarted after a code change. A running session keeps the old program in memory.

## Possible improvements

- A third provider, or a local model, behind the same `create_llm` function
- More skills, still as folders rather than new Python tools
- A `--verbose` flag that prints the full `AgentResponse` for debugging

The requirement checklist, five presentation prompts, and a short spoken explanation are in `SUBMISSION.md`.
