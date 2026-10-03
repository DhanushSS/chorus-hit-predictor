"""Keep original V1 tests intact while allowing fast/integration selections."""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name=='test_project.py': item.add_marker(pytest.mark.baseline)
        if item.path.name=='test_audio_v2.py' or item.name in {
            'test_audio_features_match_schema_and_allow_inference','test_segment_boundaries_and_invalid_audio',
            'test_demo_runs_and_predicts','test_app_run_switch_uses_matching_artifacts','test_smoke_and_checkpoint_resume'}:
            item.add_marker(pytest.mark.integration)
