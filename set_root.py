import os
import sys

# Step 1: Get the path of the file that imported this module
try:
    caller_path = os.path.abspath(sys._getframe(1).f_globals['__file__'])
except (AttributeError, ValueError):
    caller_path = os.path.abspath(__file__)

# Step 2: Detect the project root by walking up until 'set_root.py' is found
def find_project_root(start_path):
    current = start_path
    while current != os.path.dirname(current):
        if os.path.isfile(os.path.join(current, 'set_root.py')):
            return current
        current = os.path.dirname(current)
    return start_path  # fallback

project_root = find_project_root(os.path.dirname(caller_path))

# Step 3: Change working directory and fix imports
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Step 4: Optional log
print(f"[set_root] Working directory and sys.path set to: {project_root}")
