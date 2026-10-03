"""The dataset is the thing every acceptance number is measured against.

Spec section 31 requires the app show exactly 440 'Other' rows for the provided
file. If the dataset silently changes shape, every downstream figure moves and
nothing else would notice — column validation checks that a column is PRESENT,
not that the data behind it is the data the criteria were written for.
"""

from src.config import GROUND_TRUTH_COLUMNS, REQUIRED_COLUMNS
import pytest

from src.config import DEFAULT_DATASET
from src.data_io import load_dataset


def test_a_workbook_without_the_preferred_sheet_still_loads(tmp_path):
    """Uploaded workbooks name their sheet anything; the first one is read."""
    import pandas as pd

    from src.config import DATASET_SHEET

    frame = pd.read_excel(DEFAULT_DATASET, sheet_name=DATASET_SHEET, dtype=str).head(20)
    path = tmp_path / "mine.xlsx"
    frame.to_excel(path, index=False, sheet_name="Sheet1")
    loaded = load_dataset(path)
    assert loaded.report.is_valid, loaded.report.errors
    assert loaded.report.total_rows == 20


def test_there_is_no_default_dataset():
    with pytest.raises(TypeError):
        load_dataset()


def test_dataset_shape_matches_acceptance_criteria():
    loaded = load_dataset(DEFAULT_DATASET)
    report = loaded.report
    assert report.is_valid, report.errors
    assert report.total_rows == 1000
    assert report.other_rows == 440, "spec section 31 requires exactly 440"
    assert report.other_pct == 44.0
    assert report.unique_skus == 16
    assert report.unique_categories == 3
    assert report.unique_vendors == 12


def test_all_contract_columns_present():
    loaded = load_dataset(DEFAULT_DATASET)
    for column in REQUIRED_COLUMNS:
        assert column in loaded.features.columns, column
    for column in GROUND_TRUTH_COLUMNS:
        assert column in loaded.ground_truth.columns, column


def test_every_other_row_has_text_to_classify():
    """No 'Other' row is empty in this dataset, so the empty-text branch needs
    its own synthetic test (see test_router) rather than relying on real data."""
    assert load_dataset(DEFAULT_DATASET).report.other_rows_with_empty_text == 0


def test_text_bearing_structured_rows_are_the_known_deferred_wedge():
    """217 non-Other rows also carry free text — deferred under premise P1 and
    recorded in TODOS.md. Asserted so the number cannot drift unnoticed."""
    assert load_dataset(DEFAULT_DATASET).report.non_other_rows_with_text == 217


def test_eligible_record_count_matches_other_rows():
    loaded = load_dataset(DEFAULT_DATASET)
    assert len(loaded.eligible_records()) == loaded.report.other_rows
