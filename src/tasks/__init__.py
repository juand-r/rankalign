"""
Task definitions module.

Import this module to register all custom tasks with the task registry.

To add a new task:
    1. Create a new file in this directory (copy from example_task.py)
    2. Implement the required functions:
       - load_data(seed, split_type, **kwargs) -> (L_train, L_test)
       - make_prompt(item, style, shots, **kwargs) -> PromptCompletion
       - get_completion(item) -> str
       - get_label(item) -> str ('yes' or 'no')
    3. Call register_task() at module level with your config
    4. Import your task module below

The import triggers the register_task() call, adding your task to the registry.
Tasks not in the registry will use the legacy if/elif code paths.

Example:
    # In src/tasks/my_custom_task.py:
    from task_registry import register_task
    
    def load_data(...): ...
    def make_prompt(...): ...
    def get_completion(item): return " " + item['answer']
    def get_label(item): return 'yes' if item['correct'] else 'no'
    
    register_task({
        'name': 'my-custom-task',
        'load_data': load_data,
        'make_prompt': make_prompt,
        'get_completion': get_completion,
        'get_label': get_label,
    })
    
    # Then add here:
    from . import my_custom_task
"""

# Import task modules to trigger their registration.
# Add new task imports below as they are created.
#
# Example:
# from . import my_custom_task

# Note: example_task.py is a template and intentionally NOT imported here
# because it contains placeholder implementations that would fail.

# Registered tasks:
from . import hypernym2        # Replica of legacy 'hypernym' for testing registry
from . import ksat             # 2-SAT and 3-SAT tasks
# from . import hypernym_custom  # DEPRECATED - superseded by hypernym_hyponyms
from . import hyponym          # Hyponym task (reverse of hypernym)
from . import hypernym_hyponyms  # Auto-registered hypernym tasks for each hyponym dataset
from . import hypernym_concat   # Concatenated hypernym task (balanced sampling from all datasets)
from . import hypernym_concat_subset  # Subset: bananas to dogs only
from . import hypernym_concat_subset_v2  # Subset v2: bananas to dogs (fixed/grammar-corrected)
from . import ifeval_per_prompt  # Auto-registered IFEval tasks for each prompt
from . import ifeval_concat      # Concatenated IFEval task (all prompts)
from . import ambigqa_v0         # AmbigQA + PlausibleQA combined v0 + per-question tasks
from . import ambigqa_v1         # AmbigQA v1: with_negatives train + per-question test tasks
from . import plausibleqa_v0     # PlausibleQA per-question tasks
from . import rosch              # Rosch-1975 category membership eval tasks
from . import codecontests       # CodeContests competitive programming tasks