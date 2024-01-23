from langchain_core.prompts import ChatPromptTemplate
from enum import Enum
import json

class FlowType(Enum):
  BASE = 0
  RESTRICTION = 1
  HEAT_ICE = 2
  EXPECTATION = 3

with open("flow-templates.json", "r") as f:
  flows_json = json.loads(f.read())

class Flow:
  flows: list[str] = [template for template in flows_json]

  @classmethod
  def template(cls, question: str, relevant_documents: list[str], flow_type: FlowType):
    template_filled = cls.flows[flow_type.value].format(question=question, relevant_documents=relevant_documents)
    return template_filled