import asyncio
from abc import ABC, abstractmethod
from collections import Counter
from typing import Literal
from pydantic import create_model
from .llm_api import LLM_API, ResponseType


class Agentic_Operator(ABC):
    DEFAULT_SYSTEM_PROMPT: str = ""

    def __init__(
        self,
        name: str,
        model_name: str,
        input_token_cpm: float,
        output_token_cpm: float,
        n: int = 1,
        system_prompt: str = None,
        **llm_kwargs
    ):
        self.name = name
        self.n = n
        self._init_params = {
            "model_name": model_name,
            "input_token_cpm": input_token_cpm,
            "output_token_cpm": output_token_cpm,
            "system_prompt": system_prompt,
            **llm_kwargs
        }
        self.llm = LLM_API(
            model_name=model_name,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            input_token_cpm=input_token_cpm,
            output_token_cpm=output_token_cpm,
            **llm_kwargs
        )

    def __call__(self, *args, **kwargs):
        if self.n == 1:
            return self._call(*args, **kwargs)
        return self._call_with_consistency(*args, **kwargs)

    def _call_with_consistency(self, *args, **kwargs):
        instances = [self._clone() for _ in range(self.n)]
        async def _run():
            return await asyncio.gather(
                *[asyncio.to_thread(inst._call, *args, **kwargs) for inst in instances]
            )
        results = asyncio.run(_run())
        return self._aggregate(results)

    def _clone(self):
        return type(self)(name=self.name, n=1, **self._init_params)

    def _aggregate(self, results: list):
        return Counter(str(r) for r in results).most_common(1)[0][0]

    @abstractmethod
    def _call(self, *args, **kwargs): ...

    @property
    def metrics(self):
        return self.llm.metrics

    def reset(self):
        self.llm.reset()


class Predict(Agentic_Operator):
    def _call(self, prompt: str, response_type=ResponseType.TEXT, response_basemodel=None):
        raw = self.llm(prompt, response_type=response_type, response_basemodel=response_basemodel)
        return response_basemodel.model_validate_json(raw) if response_basemodel else raw


class CoT(Agentic_Operator):
    DEFAULT_SYSTEM_PROMPT = (
        "You are a careful reasoner. When given a problem, think through it thoroughly step by step."
    )

    def __init__(self, name, model_name, input_token_cpm, output_token_cpm, n=1, system_prompt=None, **llm_kwargs):
        llm_kwargs["keep_history"] = True
        super().__init__(name, model_name, input_token_cpm, output_token_cpm, n=n, system_prompt=system_prompt, **llm_kwargs)

    def _call(self, prompt: str, response_type=ResponseType.TEXT, response_basemodel=None):
        reasoning = self.llm(f"{prompt}\n\nThink through this carefully, step by step.")
        raw = self.llm(
            "Based on your reasoning above, what is your final answer? Be concise.",
            response_type=response_type,
            response_basemodel=response_basemodel
        )
        self.llm.reset_history()
        answer = response_basemodel.model_validate_json(raw) if response_basemodel else raw
        return reasoning, answer

    def _aggregate(self, results: list[tuple]) -> tuple:
        votes = Counter(str(answer) for _, answer in results)
        winning_str = votes.most_common(1)[0][0]
        reasoning, answer = next((r, a) for r, a in results if str(a) == winning_str)
        return reasoning, answer


class React(Agentic_Operator):
    DEFAULT_SYSTEM_PROMPT = (
        "You are a ReAct agent. At each step: reason about what you know and what you need. "
        "If you need more information, call a tool. "
        "If you have enough information, give your final answer directly without calling any tool."
    )

    def __init__(self, name, model_name, input_token_cpm, output_token_cpm, n=1, system_prompt=None, **llm_kwargs):
        llm_kwargs["keep_history"] = True
        super().__init__(name, model_name, input_token_cpm, output_token_cpm, n=n, system_prompt=system_prompt, **llm_kwargs)

    def _call(self, prompt: str, tools: list[callable], max_iterations: int = 5,
              response_type=ResponseType.TEXT, response_basemodel=None):
        result = self.llm(prompt, tools=tools)

        for _ in range(max_iterations - 1):
            if isinstance(result, str):
                break
            result = self.llm("Continue reasoning based on the tool results above.", tools=tools)

        if not isinstance(result, str):
            self.llm.reset_history()
            raise RuntimeError(f"React reached max_iterations={max_iterations} without a final answer.")

        if response_basemodel:
            raw = self.llm(
                "Based on your analysis above, provide your final answer in the required format.",
                response_type=response_type,
                response_basemodel=response_basemodel
            )
            self.llm.reset_history()
            return response_basemodel.model_validate_json(raw)

        self.llm.reset_history()
        return result


class Router(Agentic_Operator):
    DEFAULT_SYSTEM_PROMPT = (
        "You are a routing agent. Classify the input into exactly one of the provided categories. "
        "Return only the category name, nothing else."
    )

    def _call(self, prompt: str, routes: list[str] | type) -> str:
        if isinstance(routes, type):
            valid = [r.value for r in routes]
        else:
            valid = list(routes)
        RouteModel = create_model("Route", route=(Literal[tuple(valid)], ...))
        raw = self.llm(prompt, response_type=ResponseType.JSON_SCHEMA, response_basemodel=RouteModel)
        return RouteModel.model_validate_json(raw).route