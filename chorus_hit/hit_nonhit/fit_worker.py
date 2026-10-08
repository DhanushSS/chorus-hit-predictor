"""Private subprocess entrypoint; receives only trusted locally generated fit requests."""
import argparse
import time
import warnings
import joblib
from threadpoolctl import threadpool_limits


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True); p.add_argument('--response', required=True)
    a = p.parse_args(); request = joblib.load(a.request)
    began = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        warnings.simplefilter('always')
        request['model'].fit(request['X'], request['y'])
    joblib.dump({'model': request['model'], 'seconds': time.perf_counter() - began,
                 'warnings': [str(w.message) for w in caught]}, a.response)


if __name__ == '__main__':
    main()
