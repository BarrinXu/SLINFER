import os

# Try to get the environment variable PROJECT_BASE
project_base = os.environ.get("PROJECT_BASE")
if project_base is None:
    raise EnvironmentError("Environment variable PROJECT_BASE is not set!")

models_path_template = {
    'llama-2-13b': {
        'cpu': os.path.join(project_base, 'cpu_models/Llama-2-13b-chat-hf'),
        'gpu': os.path.join(project_base, 'gpu_models/Llama-2-13b-chat-hf'),
    },
    'llama-2-7b': {
        'cpu': os.path.join(project_base, 'cpu_models/Llama-2-7b-chat-hf'),
        'gpu': os.path.join(project_base, 'gpu_models/Llama-2-7b-chat-hf'),
    },
    'llama-3.2-3b': {
        'cpu': os.path.join(project_base, 'cpu_models/Llama-3.2-3B-Instruct'),
        'gpu': os.path.join(project_base, 'gpu_models/Llama-3.2-3B-Instruct'),
    },
}