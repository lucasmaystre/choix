import glob
import json
import os.path

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def iter_testcases(dtype: str):
    pattern = os.path.join(DATA_ROOT, "testcase-{}-*.json".format(dtype))
    for path in glob.glob(pattern):
        with open(path) as f:
            yield json.load(f)
