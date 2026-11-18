"""Check that the deployment files agree with the installed environment."""

import json
import re
from pathlib import Path

import gradio
import sklearn

ROOT = Path(__file__).resolve().parent.parent


def test_saved_model_files_exist():
    assert (ROOT / "models" / "no_show.joblib").is_file()
    assert (ROOT / "models" / "model_info.json").is_file()


def test_scikit_learn_version_matches_saved_model():
    info = json.loads((ROOT / "models" / "model_info.json").read_text(encoding="utf-8"))
    assert sklearn.__version__ == info["scikit_learn_version"]


def test_requirements_have_matching_scikit_learn_version():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert f"scikit-learn=={sklearn.__version__}" in requirements


def test_gradio_is_only_in_development_requirements():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    dev_requirements = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")

    assert re.search(r"(?m)^gradio\s*==", requirements) is None
    assert f"gradio=={gradio.__version__}" in dev_requirements


def test_readme_space_settings_match_gradio():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert f"sdk_version: {gradio.__version__}" in readme
    assert "sdk: gradio" in readme
    assert "app_file: app.py" in readme


def test_application_file_exists():
    assert (ROOT / "app.py").is_file()
