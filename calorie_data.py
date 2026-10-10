"""Nutrition lookup table and calorie-estimation helpers.

All calorie values are approximate kcal per 100 g, and all portions are
assumed serving sizes in grams. Edit these dictionaries to add foods or change
serving sizes. Keys must be normalized labels (lowercase, single spaces), see
normalize_label().
"""

# Approximate kcal per 100 g.
CALORIES = {
    # Food classes the default COCO-trained YOLO model can detect
    "apple": 52,
    "banana": 89,
    "orange": 47,
    "broccoli": 34,
    "carrot": 41,
    "sandwich": 250,
    "hot dog": 290,
    "pizza": 266,
    "donut": 452,
    "cake": 350,
    # Extra foods for a custom food-trained model (models/best.pt)
    "rice": 130,
    "chapati": 297,
    "idli": 146,
    "dosa": 168,
    "dal": 116,
    "curry": 150,
    "biryani": 180,
    "egg": 155,
    "french fries": 312,
    "burger": 295,
    "salad": 20,
}

# Assumed portion size in grams for one detected item.
DEFAULT_PORTION_GRAMS = {
    "apple": 180,
    "banana": 120,
    "orange": 130,
    "broccoli": 90,
    "carrot": 60,
    "sandwich": 150,
    "hot dog": 100,
    "pizza": 107,      # one slice
    "donut": 60,
    "cake": 80,        # one slice
    "rice": 180,
    "chapati": 40,
    "idli": 40,
    "dosa": 100,
    "dal": 150,
    "curry": 150,
    "biryani": 250,
    "egg": 50,
    "french fries": 117,
    "burger": 220,
    "salad": 100,
}

# Used when a food has a calorie value but no portion entry.
FALLBACK_PORTION_GRAMS = 100

# Alternative model class names that map onto a database key.
ALIASES = {
    "hotdog": "hot dog",
    "doughnut": "donut",
    "fries": "french fries",
    "roti": "chapati",
    "hamburger": "burger",
    "boiled egg": "egg",
    "white rice": "rice",
}


def normalize_label(label):
    """Lowercase, turn _/- into spaces, collapse whitespace, apply aliases."""
    text = str(label).lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    return ALIASES.get(text, text)


def estimate_calories(label):
    """Return (portion_g, kcal_per_100g, kcal) for a label, or None if unknown.

    Formula: kcal = kcal per 100 g x portion (g) / 100
    """
    key = normalize_label(label)
    if key not in CALORIES:
        return None
    per_100g = CALORIES[key]
    portion = DEFAULT_PORTION_GRAMS.get(key, FALLBACK_PORTION_GRAMS)
    return portion, per_100g, round(per_100g * portion / 100, 1)


def summarize(detections):
    """Build result rows and a total from [(label, confidence), ...].

    Unmatched labels get None for portion and calories, and they don't count
    toward the total.
    """
    rows = []
    total = 0.0
    for label, conf in detections:
        estimate = estimate_calories(label)
        if estimate is None:
            rows.append({"label": label, "confidence": conf, "portion": None,
                         "per_100g": None, "kcal": None})
        else:
            portion, per_100g, kcal = estimate
            total += kcal
            rows.append({"label": label, "confidence": conf, "portion": portion,
                         "per_100g": per_100g, "kcal": kcal})
    return rows, round(total, 1)
