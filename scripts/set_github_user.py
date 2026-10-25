"""Подставить ваш GitHub-ник вместо <ВАШ_НИК> в README и ноутбуке.

    python scripts/set_github_user.py my-nick
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER = "<ВАШ_НИК>"

if len(sys.argv) != 2:
    sys.exit(__doc__)
user = sys.argv[1].strip().lstrip("@")
for path in [ROOT / "README.md", ROOT / "notebooks" / "train_colab.ipynb"]:
    text = path.read_text(encoding="utf-8")
    if PLACEHOLDER in text:
        path.write_text(text.replace(PLACEHOLDER, user), encoding="utf-8")
        print(f"обновлён {path.relative_to(ROOT)}")
