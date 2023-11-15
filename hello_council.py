import os

from council.agents import Agent
from council.chains import Chain
from council.controllers import LLMController
from council.evaluators import LLMEvaluator
from council.filters import BasicFilter
from council.llm import OpenAILLM
from council.skills import LLMSkill
import dotenv


dotenv.load_dotenv()
print(os.getenv("OPENAI_API_KEY", None) is not None)

openai_llm = OpenAILLM.from_env()

hw_prompt = "You are responding to every prompt with a short poem titled hello world"
hw_skill = LLMSkill(llm=openai_llm, system_prompt=hw_prompt)
hw_chain = Chain(name="Hello World", description="Answers with a poem titled Hello World", runners=[hw_skill])

em_prompt = "You are responding to every prompt with an emoji that best addresses the question asked or statement made"
em_skill = LLMSkill(llm=openai_llm, system_prompt=em_prompt)
em_chain = Chain(name="Emoji", description="Responds to every prompt with an emoji that best fits the prompt",
                 runners=[em_skill])

controller = LLMController(chains=[hw_chain, em_chain], llm=openai_llm, response_threshold=5)
evaluator = LLMEvaluator(llm=openai_llm)

agent = Agent(controller=controller, evaluator=evaluator, filter=BasicFilter())

result = agent.execute_from_user_message("Hello world?!")
print(result.best_message.message)

result = agent.execute_from_user_message("Represent with emojis, council a multi-agent framework")
print(result.best_message.message)
