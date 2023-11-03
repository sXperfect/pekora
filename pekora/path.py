import os
import sys
from git.repo import Repo

repository = Repo('.', search_parent_directories=True)
ROOT = repository.working_tree_dir

assert isinstance(ROOT, str)
sys.path.append(ROOT)