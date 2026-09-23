"""Developer Assistant AI Agent.

Package layout:
- app.config  -> loads settings from environment variables
- app.agent   -> LLM setup and the autonomous tool-calling agent
- app.tools   -> custom LangChain tools (calculator, file reader)
- app.models  -> Pydantic schemas for structured data
- app.main    -> interactive CLI entry point
"""
