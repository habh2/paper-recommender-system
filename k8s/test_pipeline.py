import importlib.util
import sys
import tempfile
import yaml
import pytest
from pathlib import Path

pytestmark = pytest.mark.unit

# Import pipeline module without requiring k8s to be a package
spec = importlib.util.spec_from_file_location("pipeline", Path(__file__).parent / "pipeline.py")
pipeline_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_mod)


@pytest.fixture(scope="module")
def data_yaml():
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        path = f.name
    pipeline_mod.compiler.Compiler().compile(pipeline_mod.data_pipeline, path)
    with open(path) as f:
        docs = list(yaml.safe_load_all(f))
    return next(d for d in docs if d and "pipelineInfo" in d)


@pytest.fixture(scope="module")
def preference_yaml():
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        path = f.name
    pipeline_mod.compiler.Compiler().compile(pipeline_mod.preference_pipeline, path)
    with open(path) as f:
        docs = list(yaml.safe_load_all(f))
    return next(d for d in docs if d and "pipelineInfo" in d)


def _component_names(compiled_yaml):
    components = compiled_yaml.get("components", {})
    return set(components.keys())


def test_data_pipeline_name(data_yaml):
    assert data_yaml["pipelineInfo"]["name"] == "data-pipeline"


def test_data_pipeline_has_four_steps(data_yaml):
    names = _component_names(data_yaml)
    assert len(names) == 4, f"Expected 4 components, got {len(names)}: {names}"


def test_data_pipeline_steps(data_yaml):
    names = _component_names(data_yaml)
    assert "comp-ingest" in names
    assert "comp-embed" in names
    assert "comp-train-topic" in names
    assert "comp-compute-dist" in names


def test_preference_pipeline_name(preference_yaml):
    assert preference_yaml["pipelineInfo"]["name"] == "preference-pipeline"


def test_preference_pipeline_has_three_steps(preference_yaml):
    names = _component_names(preference_yaml)
    assert len(names) == 3, f"Expected 3 components, got {len(names)}: {names}"


def test_preference_pipeline_steps(preference_yaml):
    names = _component_names(preference_yaml)
    assert "comp-train-pref" in names
    assert "comp-evaluate" in names
    assert "comp-restart-app" in names
