"""clinician/common/config.py -- config loading shared by every rater."""

import os
import re

import yaml

CLINICIAN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(CLINICIAN)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def resolve(path):
    """${VAR} and ${VAR:-default}; relative results are relative to the repo.
    The same file then works on the laptop and on Bouchet: env.sh sets the
    variables there, the defaults are the laptop paths."""
    def sub(m):
        val = os.environ.get(m.group(1))
        if val:
            return val
        if m.group(2) is not None:
            return m.group(2)
        raise SystemExit(f"config: ${{{m.group(1)}}} is unset and has no default")
    out = os.path.expanduser(_ENV_RE.sub(sub, str(path)))
    return out if os.path.isabs(out) else os.path.normpath(os.path.join(REPO, out))


def load(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
