"""dataset._store_once reads each input-HDF table once per process.

clear_iteration_cache drops orca's cache every year; without this the cached
tables and injectables that read the input HDF re-read it annually. Run from the
repository root:
    PYTHONPATH=. python scripts/testing/test_store_once.py
"""
import os
import sys

import orca
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# dataset.py loads the input HDF on import and writes a schema snapshot there
orca.add_injectable("data_out_dir", os.path.join("/tmp", "test_store_once"))
os.makedirs(orca.get_injectable("data_out_dir"), exist_ok=True)


class CountingStore:
    """Stands in for the HDFStore, counting reads per key."""

    def __init__(self, frames):
        self.frames, self.reads = frames, {}

    def __getitem__(self, key):
        self.reads[key] = self.reads.get(key, 0) + 1
        return self.frames[key]


def t_second_read_is_served_from_the_cache():
    import dataset
    dataset._store_once_cache.clear()
    st = CountingStore({"buildings": pd.DataFrame({"x": [1, 2]})})
    for _ in range(30):                      # one per simulated year
        dataset._store_once(st, "buildings")
    assert st.reads["buildings"] == 1, st.reads
    print("  PASS 30 calls, 1 read of the input HDF")


def t_each_key_cached_separately():
    import dataset
    dataset._store_once_cache.clear()
    st = CountingStore({"a": pd.DataFrame({"x": [1]}), "b": pd.DataFrame({"y": [2]})})
    for _ in range(5):
        dataset._store_once(st, "a")
        dataset._store_once(st, "b")
    assert st.reads == {"a": 1, "b": 1}, st.reads
    print("  PASS keys cached independently")


def t_returns_the_frame_unchanged():
    import dataset
    dataset._store_once_cache.clear()
    df = pd.DataFrame({"x": [1, 2, 3]})
    st = CountingStore({"k": df})
    assert dataset._store_once(st, "k") is df
    assert dataset._store_once(st, "k") is df
    print("  PASS the frame itself is returned, not a copy")


def t_btype_owner_share_reads_buildings_once():
    import dataset
    dataset._store_once_cache.clear()
    b = pd.DataFrame({"building_type_id": [81, 81, 83],
                      "residential_units": [10, 10, 20],
                      "owner_units": [8, 6, 5]})
    st = CountingStore({"buildings": b})
    fn = getattr(dataset.btype_owner_share, "_func", dataset.btype_owner_share)
    out = [fn(st) for _ in range(30)]
    assert st.reads["buildings"] == 1, st.reads
    assert out[0] == out[-1]
    assert abs(out[0][81] - 14 / 20) < 1e-9 and abs(out[0][83] - 5 / 20) < 1e-9, out[0]
    print("  PASS btype_owner_share: 30 calls, 1 read, shares %s"
          % {k: round(v, 3) for k, v in out[0].items()})


def t_every_store_reader_goes_through_it():
    """No cached definition in dataset.py still reads a never-replaced table directly."""
    import re
    src = open(os.path.join(ROOT, "dataset.py")).read()
    never_replaced = ("parcel_maz_crossing_shares", "points_of_interest_by_category")
    bad = [k for k in never_replaced if re.search(r'store\["%s"\]' % k, src)]
    assert not bad, "still read directly: %s" % bad
    print("  PASS no direct store read left for the never-replaced tables")


def main():
    tests = [t_second_read_is_served_from_the_cache, t_each_key_cached_separately,
             t_returns_the_frame_unchanged, t_btype_owner_share_reads_buildings_once,
             t_every_store_reader_goes_through_it]
    fails = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print("  FAIL %s: %s" % (fn.__name__, e))
    print("%d of %d passed" % (len(tests) - fails, len(tests)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
