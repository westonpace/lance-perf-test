import os
import random
import shutil

import lmdb
import pyarrow as pa
import lance

# 65M, 1183 keys
LMDB_LOCAL_PATH = "../lmdb_dataset"
LANCE_LOCAL_PATH = "../lance_dataset"

# COMPRESSION = {"lance-encoding:compression": "zstd"}
COMPRESSION = {}
SCHEMA = pa.schema([
    pa.field("key", pa.string(), metadata={**COMPRESSION}),
    pa.field("value", pa.large_binary(), metadata={**COMPRESSION}),
])
STORAGE_OPTIONS = {}

PERF_RUNS = 1
PERF_RANDOM_RANGE = 1183
PERF_RANDOM_COUNT = 1000

LMDB_MMAP_SIZE = 100 << 20  # 100M


def rm_dir(path):
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
        except Exception as e:
            print(f"remove {path}: {e}")


def gen_key(key):
    return f"{key:032d}"


def lmdb_to_dict():
    data = {
        "key": [],
        "value": [],
    }

    env = lmdb.open(LMDB_LOCAL_PATH, readonly=True, lock=False)
    reader = env.begin(write=False)
    for k, v in reader.cursor():
        data['key'].append(k.decode())
        data['value'].append(v)

    return data


def create_lance(data):
    rm_dir(LANCE_LOCAL_PATH)

    table = pa.Table.from_pydict(data, schema=SCHEMA)
    ds = lance.write_dataset(table, LANCE_LOCAL_PATH, schema=SCHEMA)
    print(f"keys: {ds.count_rows()}")


def rewrite_dataset():
    data = lmdb_to_dict()
    print(f"keys: {len(data['key'])}")

    create_lance(data)


def perf_test_lmdb():
    env = lmdb.open(LMDB_LOCAL_PATH, readonly=True, lock=False, map_size=LMDB_MMAP_SIZE)
    with env.begin(write=False) as reader:
        for _ in range(PERF_RANDOM_COUNT):
            key = gen_key(random.randrange(0, PERF_RANDOM_RANGE)).encode()
            val = reader.get(key)
            assert val
            len(val)
    env.close()


def perf_test_lance_take():
    ds = lance.dataset(LANCE_LOCAL_PATH, storage_options=STORAGE_OPTIONS, index_cache_size=1 << 30)
    for _ in range(PERF_RANDOM_COUNT):
        filtered = ds.take(indices=[random.randrange(0, PERF_RANDOM_RANGE)], columns=["value"])
        assert filtered.num_rows == 1
        len(filtered['value'][0].as_py())


def perf_test():
    import timeit
    token = timeit.timeit(perf_test_lmdb, number=PERF_RUNS)
    print(f"perf lmdb cost: {token}")
    token = timeit.timeit(perf_test_lance_take, number=PERF_RUNS)
    print(f"perf lance take cost: {token}")


if __name__ == '__main__':
    rewrite_dataset()
    perf_test()
