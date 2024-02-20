from langchain_core.prompts import ChatPromptTemplate
from enum import Enum
import json
import os

class FlowType(Enum):
  ROOT = "root"
  BASE = "base"
  RESTRICTION = "restriction"
  HEAT_ICE = "heat_ice"
  EXPECTATION = "expectation"

with open(os.path.join(os.path.dirname(__file__),"flow-templates.json"), "r") as f:
  flows_json = json.loads(f.read())

class Flow:
  flows: map = flows_json

  @classmethod
  def root(cls) -> str:
    return cls.flows[FlowType.ROOT.value]

  @classmethod
  def template(cls, injury: str, injury_location:str, flow_type: FlowType):
    template_filled = cls.flows[flow_type.value].format(injury=injury, location=injury_location)
    return template_filled