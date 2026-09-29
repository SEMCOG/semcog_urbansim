"""Shared naming convention for optional REPM city fixed-effect features."""

CITY_FEATURE_PREFIX = "city_is_"


def city_feature_name(city_id):
    return f"{CITY_FEATURE_PREFIX}{int(city_id)}"


def city_id_from_feature(feature_name):
    if not feature_name.startswith(CITY_FEATURE_PREFIX):
        return None
    suffix = feature_name[len(CITY_FEATURE_PREFIX):]
    return int(suffix) if suffix.isdigit() else None
