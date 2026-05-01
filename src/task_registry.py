"""
Task Registry for extensible task definitions.

This module provides a registry pattern for adding new tasks without modifying
existing code. New tasks register themselves here; existing tasks continue to
use legacy if/elif chains (which remain unchanged).

Usage for new tasks:
    1. Create a new file in src/tasks/ (copy from example_task.py)
    2. Implement: load_data, make_prompt, get_completion, get_label
    3. Call register_task() with your configuration
    4. Import your task module in src/tasks/__init__.py

The registry is checked first; if a task is not registered, the legacy code
paths are used. This ensures backward compatibility.
"""

from typing import Dict, Callable, Any, Optional, List


# The registry - empty by default, so all existing tasks use legacy paths
TASK_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_task(config: Dict[str, Any]) -> None:
    """
    Register a new task configuration.
    
    Required fields:
        name: str
            Task identifier (used in --task argument)
        
        load_data: Callable(seed: int, split_type: str, **kwargs) -> (L_train, L_test)
            Function to load and split data. Should return train/test lists.
        
        make_prompt: Callable(item, style: str, shots: str, **kwargs) -> PromptCompletion
            Function to create prompts. style is 'generator' or 'discriminator'.
            shots is 'zero' or 'few'. Returns namedtuple with .prompt and .completion.
        
        get_completion: Callable(item) -> str
            Extract generator completion text from data item.
            This is the text the generator should produce (e.g., the answer).
            Should include leading space if needed for tokenization.
        
        get_label: Callable(item) -> str
            Extract binary label from data item.
            Must return 'yes' or 'no' (lowercase).
    
    Optional fields:
        get_indicator: Callable(item) -> int
            Return 1 for positive examples, 0 for negative.
            Defaults to: 1 if get_label(item) == 'yes' else 0
        
        batch_size: Dict[str, int]
            Batch sizes for training. Format: {'with_ref': N, 'without_ref': M}
            Defaults to {'with_ref': 1, 'without_ref': 2}
        
        supports_negative_sampling: bool
            Whether task supports --sample_negative flag.
            Defaults to False.
        
        filter_positive: Callable(item) -> bool
            Filter function for positive examples only.
            Defaults to: get_label(item) == 'yes'
        
        supports_split_types: List[str]
            List of supported split_type values.
            Defaults to ['random'].

        make_negated_prompt: Callable(item, task, make_prompt, gen_shots) -> (neg_prompt, completion)
            Optional callback for --neg-typicality prompt construction.
            If omitted, callers can fall back to legacy branching logic.

        csv_header: List[str]
            Optional ordered list of CSV column names for --save-scores-csv exports.

        csv_row_builder: Callable(...) -> List[Any]
            Optional callback to build a CSV row from scored item fields.
            Used by generic CSV save logic for newer task families.
    
    Example:
        register_task({
            'name': 'my-task',
            'load_data': my_load_function,
            'make_prompt': my_prompt_function,
            'get_completion': lambda item: " " + item['answer'],
            'get_label': lambda item: 'yes' if item['correct'] else 'no',
        })
    """
    name = config.get('name')
    if name is None:
        raise ValueError("Task config missing required field: 'name'")
    
    # Validate required fields
    required = ['name', 'load_data', 'make_prompt', 'get_completion', 'get_label']
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"Task '{name}' config missing required fields: {missing}")
    
    # Set defaults for optional fields
    if 'get_indicator' not in config:
        get_label = config['get_label']
        config['get_indicator'] = lambda item, _gl=get_label: 1 if _gl(item) == 'yes' else 0
    
    if 'batch_size' not in config:
        config['batch_size'] = {'with_ref': 1, 'without_ref': 2}
    
    if 'supports_negative_sampling' not in config:
        config['supports_negative_sampling'] = False
    
    if 'filter_positive' not in config:
        get_label = config['get_label']
        config['filter_positive'] = lambda item, _gl=get_label: _gl(item) == 'yes'
    
    if 'supports_split_types' not in config:
        config['supports_split_types'] = ['random']
    
    TASK_REGISTRY[name] = config
    print(f"[task_registry] Registered task: {name}")


def get_task(name: str) -> Optional[Dict[str, Any]]:
    """
    Get task configuration by name.
    
    Args:
        name: Task identifier
    
    Returns:
        Task configuration dict if registered, None otherwise.
        None triggers legacy fallback in calling code.
    """
    return TASK_REGISTRY.get(name, None)


def list_registered_tasks() -> List[str]:
    """
    Return list of all registered task names.
    
    Useful for dynamically building argparse choices.
    """
    return list(TASK_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """
    Check if a task is registered.
    
    Args:
        name: Task identifier
    
    Returns:
        True if task is in registry, False otherwise.
    """
    return name in TASK_REGISTRY


def get_all_task_names(legacy_tasks: List[str]) -> List[str]:
    """
    Get combined list of legacy and registered task names.
    
    Useful for argparse choices that include both old and new tasks.
    
    Args:
        legacy_tasks: List of task names handled by legacy if/elif chains
    
    Returns:
        Combined list with legacy tasks first, then registered tasks
    """
    registered = list_registered_tasks()
    # Avoid duplicates if a legacy task gets registered
    new_tasks = [t for t in registered if t not in legacy_tasks]
    return legacy_tasks + new_tasks

