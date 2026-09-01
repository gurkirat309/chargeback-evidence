"""
Live demo: an AI assistant answering a merchant by calling RokdaDaav's MCP tools.

Nothing here is scripted to call a specific tool — a live Groq model is given the
RokdaDaav tool list and DECIDES on its own which to call. We just print what it
does. Run:  python demo/mcp_agent_demo.py
Requires GROQ_API_KEY in .env and a built pipeline (see RUNBOOK.md).
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
from groq import Groq
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession

MODEL = "openai/gpt-oss-120b"
PARAMS = StdioServerParameters(command=sys.executable,
                               args=[str(ROOT / "src" / "mcp_server.py")])
groq = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM = ("You are a merchant's chargeback risk assistant. You have no knowledge of "
          "specific disputes yourself — you MUST use the RokdaDaav tools to look "
          "things up and decide. Be concise and end with a clear recommendation in "
          "rupees.")


def to_groq_tools(mcp_tools):
    out = []
    for t in mcp_tools:
        schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", {})
        out.append({"type": "function", "function": {
            "name": t.name, "description": (t.description or "")[:300],
            "parameters": schema}})
    return out


async def ask(question, session, tools):
    print("\n" + "─" * 74 + f"\n🧑  MERCHANT:  {question}\n" + "─" * 74)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": question}]
    for _ in range(6):
        resp = groq.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, tool_choice="auto",
            temperature=0.2, max_tokens=1500, reasoning_effort="low")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            print(f"\n🤖  ASSISTANT:  {msg.content.strip()}\n")
            return
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            print(f"\n   🤖→🔧  the AI calls RokdaDaav tool: {tc.function.name}("
                  f"{', '.join(f'{k}={v}' for k, v in args.items())})")
            result = await session.call_tool(tc.function.name, args)
            text = result.content[0].text
            print(f"   🔧→🤖  RokdaDaav returns: "
                  f"{text if len(text) < 500 else text[:500] + '…'}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": text})


async def main():
    async with stdio_client(PARAMS) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools = to_groq_tools((await session.list_tools()).tools)
            print("Connected to RokdaDaav MCP server —",
                  ", ".join(t["function"]["name"] for t in tools))
            await ask("A customer just disputed order DSP001561. Should I fight it, "
                      "and what's the expected value?", session, tools)
            await ask("What about DSP000864 — worth fighting?", session, tools)


if __name__ == "__main__":
    asyncio.run(main())
