import os
import json
import inspect
from typing import get_type_hints
from groq import Groq, AsyncGroq, RateLimitError
from dotenv import load_dotenv
from enum import StrEnum
from pydantic import BaseModel
from dataclasses import dataclass, field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


class ResponseType(StrEnum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class CompactionStrategy(StrEnum):
    SLIDING_WINDOW = "sliding_window"
    SUMMARIZE = "summarize"


@dataclass
class Metrics:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_api_time: float = 0.0
    previous_response_time: float = 0.0
    total_number_of_calls: int = 0
    total_tool_calls: int = 0
    total_retries: int = 0
    total_compactions: int = 0
    previous_prompt_tokens: int = 0
    unnatural_stop_prompts: list = field(default_factory=list)


class LLM_API:

    def __init__(
            self,
            model_name: str,
            system_prompt: str,
            input_token_cpm: float,
            output_token_cpm: float,
            keep_history: bool = False,
            max_retries: int = 5,
            min_wait: float = 4,
            max_wait: float = 60,
            max_context_tokens: int = None,
            max_output_tokens: int = None,
            compaction_strategy: CompactionStrategy = CompactionStrategy.SLIDING_WINDOW,
            compaction_threshold: float = 0.8,
            dotenv_path: str = "/home/anishsan/Documents/GitHub/MetaOS/.env"
            ):
        load_dotenv(dotenv_path=dotenv_path)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.previous_response = None
        self.input_token_cpm = input_token_cpm
        self.output_token_cpm = output_token_cpm
        self.keep_history = keep_history
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.compaction_strategy = compaction_strategy
        self.compaction_threshold = compaction_threshold
        self.metrics = Metrics()
        self.history_token_pairs: list[tuple[int, int]] = []
        self._prev_completion_tokens: int = 0

        retry_config = dict(
            retry=retry_if_exception_type(RateLimitError),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            stop=stop_after_attempt(max_retries),
            before_sleep=lambda _: setattr(self.metrics, 'total_retries', self.metrics.total_retries + 1)
        )
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self._create_completion = retry(**retry_config)(self.client.chat.completions.create)
        self.async_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self._async_create_completion = retry(**retry_config)(self.async_client.chat.completions.create)

        self.TYPE_MAP = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object"}
        if self.system_prompt:
            self.history = [
                {"role": "system", "content": self.system_prompt}
            ]

    def __call__(
            self,
            prompt: str,
            response_type: ResponseType = ResponseType.TEXT,
            response_basemodel: BaseModel = None,
            tools: list[callable] = None,
            tool_choice: str = "auto"
    ) -> str | dict:
        query = self.history + [{"role": "user", "content": prompt}]

        if tools:
            tool_schemas = self._generate_tool_schema(tools)
            available = {fn.__name__: fn for fn in tools}
            response = self._create_completion(
                model=self.model_name,
                messages=query,
                tools=tool_schemas,
                tool_choice=tool_choice,
**({"max_tokens": self.max_output_tokens} if self.max_output_tokens else {})
            )
            self.callback(response, prompt)
            if self._should_compact():
                self._compact()
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                results = {}
                for tc in tool_calls:
                    fn = available[tc.function.name]
                    args = json.loads(tc.function.arguments)
                    results[tc.function.name] = fn(**args)
                    self.metrics.total_tool_calls += 1
                return results
            return response.choices[0].message.content

        if response_type == ResponseType.JSON_SCHEMA and response_basemodel:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_basemodel.__name__.lower(),
                    "schema": response_basemodel.model_json_schema()
                }
            }
        else:
            response_format = {"type": response_type}

        response = self._create_completion(
            model=self.model_name,
            messages=query,
response_format=response_format,
            **({"max_tokens": self.max_output_tokens} if self.max_output_tokens else {})
        )

        if response.choices[0].finish_reason in ["length", "content_filter"]:
            self.metrics.unnatural_stop_prompts.append({
                "prompt": prompt,
                "finish_reason": response.choices[0].finish_reason,
                "response": response.choices[0].message
            })

        self.previous_response = response.choices[0].message
        self.callback(response, prompt)
        if self._should_compact():
            self._compact()
        return response.choices[0].message.content

    async def acall(
            self,
            prompt: str,
            response_type: ResponseType = ResponseType.TEXT,
            response_basemodel: BaseModel = None,
            tools: list[callable] = None,
            tool_choice: str = "auto"
    ) -> str | dict:
        query = self.history + [{"role": "user", "content": prompt}]

        if tools:
            tool_schemas = self._generate_tool_schema(tools)
            available = {fn.__name__: fn for fn in tools}
            response = await self._async_create_completion(
                model=self.model_name,
                messages=query,
                tools=tool_schemas,
                tool_choice=tool_choice,
**({"max_tokens": self.max_output_tokens} if self.max_output_tokens else {})
            )
            self.callback(response, prompt)
            if self._should_compact():
                await self._async_compact()
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                results = {}
                for tc in tool_calls:
                    fn = available[tc.function.name]
                    args = json.loads(tc.function.arguments)
                    results[tc.function.name] = fn(**args)
                    self.metrics.total_tool_calls += 1
                return results
            return response.choices[0].message.content

        if response_type == ResponseType.JSON_SCHEMA and response_basemodel:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_basemodel.__name__.lower(),
                    "schema": response_basemodel.model_json_schema()
                }
            }
        else:
            response_format = {"type": response_type}

        response = await self._async_create_completion(
            model=self.model_name,
            messages=query,
response_format=response_format,
            **({"max_tokens": self.max_output_tokens} if self.max_output_tokens else {})
        )

        if response.choices[0].finish_reason in ["length", "content_filter"]:
            self.metrics.unnatural_stop_prompts.append({
                "prompt": prompt,
                "finish_reason": response.choices[0].finish_reason,
                "response": response.choices[0].message
            })

        self.previous_response = response.choices[0].message
        self.callback(response, prompt)
        if self._should_compact():
            await self._async_compact()
        return response.choices[0].message.content

    def callback(self, response, prompt: str):
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        self.metrics.total_input_tokens += input_tokens
        self.metrics.total_output_tokens += output_tokens
        self.metrics.input_cost += input_tokens * self.input_token_cpm / 1_000_000
        self.metrics.output_cost += output_tokens * self.output_token_cpm / 1_000_000

        self.metrics.previous_response_time = response.usage.completion_time
        self.metrics.total_api_time += self.metrics.previous_response_time
        self.metrics.total_number_of_calls += 1

        if self.keep_history:
            user_tokens = max(0, input_tokens - self.metrics.previous_prompt_tokens - self._prev_completion_tokens)
            self.history.append({"role": "user", "content": prompt})
            self.history.append(response.choices[0].message)
            self.history_token_pairs.append((user_tokens, output_tokens))

        self.metrics.previous_prompt_tokens = input_tokens
        self._prev_completion_tokens = output_tokens

    def _should_compact(self) -> bool:
        return (
            self.max_context_tokens is not None
            and self.keep_history
            and self.metrics.previous_prompt_tokens >= self.max_context_tokens * self.compaction_threshold
        )

    def _compact(self):
        if self.compaction_strategy == CompactionStrategy.SLIDING_WINDOW:
            self._compact_sliding_window()
        elif self.compaction_strategy == CompactionStrategy.SUMMARIZE:
            self._compact_summarize()
        self.metrics.total_compactions += 1

    async def _async_compact(self):
        if self.compaction_strategy == CompactionStrategy.SLIDING_WINDOW:
            self._compact_sliding_window()
        elif self.compaction_strategy == CompactionStrategy.SUMMARIZE:
            await self._async_compact_summarize()
        self.metrics.total_compactions += 1

    def _compact_sliding_window(self):
        system_msgs = [m for m in self.history if m.get("role") == "system"]
        turns = [m for m in self.history if m.get("role") != "system"]
        if len(self.history_token_pairs) <= 1:
            return

        target = int(self.max_context_tokens * 0.6)
        to_free = self.metrics.previous_prompt_tokens - target

        freed = 0
        pairs_to_drop = 0
        for user_t, asst_t in self.history_token_pairs:
            if freed >= to_free:
                break
            freed += user_t + asst_t
            pairs_to_drop += 1

        pairs_to_drop = min(pairs_to_drop, len(self.history_token_pairs) - 1)
        self.history = system_msgs + turns[pairs_to_drop * 2:]
        self.history_token_pairs = self.history_token_pairs[pairs_to_drop:]

    def _compact_summarize(self):
        system_msgs = [m for m in self.history if m.get("role") == "system"]
        turns = [m for m in self.history if m.get("role") != "system"]
        if len(self.history_token_pairs) < 4:
            return

        midpoint = len(self.history_token_pairs) // 2
        to_summarize = turns[:midpoint * 2]
        to_keep = turns[midpoint * 2:]

        convo_text = "\n".join(
            f"{m['role'].upper()}: {m['content'] if isinstance(m.get('content'), str) else '[non-text]'}"
            for m in to_summarize
        )
        summary_response = self._create_completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Summarize the following conversation concisely, preserving all key facts, decisions, and context."},
                {"role": "user", "content": convo_text}
            ]
        )
        summary = summary_response.choices[0].message.content
        summary_msg = {"role": "system", "content": f"[Summary of earlier conversation]\n{summary}"}
        self.history = system_msgs + [summary_msg] + to_keep
        self.history_token_pairs = self.history_token_pairs[midpoint:]

    async def _async_compact_summarize(self):
        system_msgs = [m for m in self.history if m.get("role") == "system"]
        turns = [m for m in self.history if m.get("role") != "system"]
        if len(self.history_token_pairs) < 4:
            return

        midpoint = len(self.history_token_pairs) // 2
        to_summarize = turns[:midpoint * 2]
        to_keep = turns[midpoint * 2:]

        convo_text = "\n".join(
            f"{m['role'].upper()}: {m['content'] if isinstance(m.get('content'), str) else '[non-text]'}"
            for m in to_summarize
        )
        summary_response = await self._async_create_completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "Summarize the following conversation concisely, preserving all key facts, decisions, and context."},
                {"role": "user", "content": convo_text}
            ]
        )
        summary = summary_response.choices[0].message.content
        summary_msg = {"role": "system", "content": f"[Summary of earlier conversation]\n{summary}"}
        self.history = system_msgs + [summary_msg] + to_keep
        self.history_token_pairs = self.history_token_pairs[midpoint:]

    def __str__(self):
        return (f"Total Input Tokens: {self.metrics.total_input_tokens}\n"
                f"Total Output Tokens: {self.metrics.total_output_tokens}\n"
                f"Total Input Cost: ${self.metrics.input_cost:.4f}\n"
                f"Total Output Cost: ${self.metrics.output_cost:.4f}\n"
                f"Total API Time: {self.metrics.total_api_time:.2f} seconds\n"
                f"Previous Response Time: {self.metrics.previous_response_time:.2f} seconds\n"
                f"Total Number of Calls: {self.metrics.total_number_of_calls}\n"
                f"Previous Prompt Tokens: {self.metrics.previous_prompt_tokens}\n"
                f"Total Tool Calls: {self.metrics.total_tool_calls}\n"
                f"Total Retries: {self.metrics.total_retries}\n"
                f"Total Compactions: {self.metrics.total_compactions}\n"
                f"Unnatural Stop Prompts: {len(self.metrics.unnatural_stop_prompts)}")

    def reset(self):
        self.metrics = Metrics()
        self.previous_response = None
        self.history_token_pairs = []
        self._prev_completion_tokens = 0
        if self.system_prompt:
            self.history = [
                {"role": "system", "content": self.system_prompt}
            ]

    def _generate_tool_schema(self, tools: list[callable]) -> list[dict]:
        schema = []
        for tool in tools:
            sig = inspect.signature(tool)
            hints = get_type_hints(tool)
            properties = {}
            required = []
            for name, param in sig.parameters.items():
                json_type = self.TYPE_MAP.get(hints.get(name), "string")
                properties[name] = {"type": json_type, "description": name}
                if param.default is inspect.Parameter.empty:
                    required.append(name)
            schema.append({
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": tool.__doc__ or tool.__name__,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required
                    }
                }
            })
        return schema
