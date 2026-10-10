import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calorie_data import (CALORIES, DEFAULT_PORTION_GRAMS, estimate_calories,
                          normalize_label, summarize)


def test_rice_example_from_documentation():
    # 130 kcal/100 g x 180 g / 100 = 234 kcal
    assert estimate_calories("rice") == (180, 130, 234.0)


def test_normalize_label_and_aliases():
    assert normalize_label("  Hot_Dog ") == "hot dog"
    assert normalize_label("hotdog") == "hot dog"
    assert normalize_label("Doughnut") == "donut"
    assert normalize_label("French-Fries") == "french fries"


def test_unknown_label_returns_none():
    assert estimate_calories("person") is None


def test_summarize_totals_only_matched_items():
    rows, total = summarize([("pizza", 0.9), ("person", 0.8), ("banana", 0.7)])
    assert [r["kcal"] for r in rows] == [284.6, None, 106.8]
    assert total == 391.4


def test_summarize_empty():
    assert summarize([]) == ([], 0.0)


def test_every_portion_has_a_calorie_value():
    assert set(DEFAULT_PORTION_GRAMS) <= set(CALORIES)


def test_coco_food_classes_are_covered():
    coco_foods = ["banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake"]
    assert all(estimate_calories(f) is not None for f in coco_foods)
