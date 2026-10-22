"""Install missing packages into the Python that runs the pipeline (IDLE, VS Code, Terminal...)."""
import importlib.util
import site
import subprocess
import sys

REQUIRED = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "statsmodels": "statsmodels",
            "sklearn": "scikit-learn", "matplotlib": "matplotlib", "openpyxl": "openpyxl", "yaml": "pyyaml", "tabulate": "tabulate"}


def ensure_packages():
    missing = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print(f"Installing missing packages {missing} into {sys.executable} ...", flush=True)
    base = [sys.executable, "-m", "pip", "install", "--quiet"]
    for extra in ([], ["--user"], ["--user", "--break-system-packages"]):
        if subprocess.call(base + extra + missing) == 0:
            user_site = site.getusersitepackages()
            if user_site not in sys.path:
                sys.path.append(user_site)
            importlib.invalidate_caches()
            return
    sys.exit(f"Automatic install failed. Run in a terminal:\n  {sys.executable} -m pip install {' '.join(missing)}")
