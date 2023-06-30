import os
import json
from typing import Union

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Set the alpaca template as default
            template_name = "alapaca"
        file_name = os.path.join("templates", f"{template_name}.json")
        if not os.path.exists(file_name):
            raise ValueError(f"Can't find {file_name}")
        with open(file_name) as file:
            self.template = json.load(file)
        if self._verbose:
            print(f"Using prompt tempalte {template_name}: {self.template['description']}")
    
    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template['prompt_input'].format(instruction=instruction, input=input)
        else:
            res = self.template['prompt_no_input'].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res
    
    def get_response(self, output: str) -> str:
        return output.split(self.template['response_split'])[1].strip()
