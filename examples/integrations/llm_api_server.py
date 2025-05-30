import asyncio
import os
import socket
import sys
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Annotated

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from langchain_ollama import ChatOllama

from browser_use import Agent, BrowserProfile, BrowserSession
from browser_use.agent.views import AgentHistoryList

load_dotenv()

DEFAULT_MODEL = 'qwen3:30b-a3b-q8_0'
LOCAL_MODEL = 'qwen2.5vl:7b-q8_0'
API_KEY = os.getenv('BROWSER_USE_API_KEY')

app = FastAPI(title='Browser Use API')

tasks: dict[str, asyncio.Task] = {}
results: dict[str, str] = {}
debug_urls: dict[str, str] = {}


def create_llm(use_local: bool = False) -> ChatOllama:
	"""Return an LLM instance configured for remote or local execution."""
	model = LOCAL_MODEL if use_local else DEFAULT_MODEL
	if use_local:
		os.environ['OLLAMA_HOST'] = os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')
	return ChatOllama(model=model, num_ctx=128000, temperature=0.0)


def get_free_port() -> int:
	"""Return a free TCP port."""
	with socket.socket() as s:
		s.bind(('', 0))
		return s.getsockname()[1]


async def _run_agent(session_id: str, task: str, use_local: bool, port: int) -> None:
	"""Execute the agent and store the result."""
	profile = BrowserProfile(headless=False, args=[f'--remote-debugging-port={port}'])
	browser_session = BrowserSession(browser_profile=profile)
	llm = create_llm(use_local)
	agent = Agent(task=task, llm=llm, browser_session=browser_session, max_failures=5, use_vision=True)
	try:
		history: AgentHistoryList = await agent.run()
		results[session_id] = history.final_result() or ''
	except Exception as e:
		results[session_id] = f'Error: {e}'
	finally:
		tasks.pop(session_id, None)


def verify_key(x_api_key: str | None) -> None:
	"""Validate the provided API key."""
	if API_KEY and x_api_key != API_KEY:
		raise HTTPException(status_code=401, detail='Invalid API key')


@app.post('/task')
async def run_task(
	prompt: str,
	local: bool = False,
	x_api_key: Annotated[str | None, Header(None)] = None,
) -> dict[str, str]:
	"""Start a new browser task and return its session ID."""
	verify_key(x_api_key)
	session_id = str(uuid.uuid4())
	port = get_free_port()
	debug_urls[session_id] = f'http://{os.getenv("PUBLIC_HOST", "localhost")}:{port}'
	tasks[session_id] = asyncio.create_task(_run_agent(session_id, prompt, local, port))
	return {'session_id': session_id, 'debug_url': debug_urls[session_id]}


@app.get('/status/{session_id}')
async def task_status(
	session_id: str,
	x_api_key: Annotated[str | None, Header(None)] = None,
) -> dict[str, str]:
	"""Return the status or result of a task."""
	verify_key(x_api_key)
	if session_id in results:
		return {'status': 'done', 'result': results[session_id], 'debug_url': debug_urls.get(session_id)}
	if session_id in tasks:
		return {'status': 'running', 'debug_url': debug_urls.get(session_id)}
	raise HTTPException(status_code=404, detail='Session not found')


if __name__ == '__main__':
	uvicorn.run(app, host='0.0.0.0', port=8000)
