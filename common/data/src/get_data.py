"""
Compatibility facade for legacy imports.

Public functions are re-exported from jv_pipeline so existing notebooks
and scripts can keep importing `common.data.src.get_data`.
"""

try:
    from .jv_pipeline import fetch_jra_data as fetch_jra_data
    from .jv_pipeline import *  # noqa: F401,F403
except ImportError:
    from jv_pipeline import fetch_jra_data as fetch_jra_data
    from jv_pipeline import *  # noqa: F401,F403

try:
    from .legacy_get_data_impl import run_today_se_ra_and_realtime_merge as run_today_se_ra_and_realtime_merge
except ImportError:
    from legacy_get_data_impl import run_today_se_ra_and_realtime_merge as run_today_se_ra_and_realtime_merge


if __name__ == "__main__":
    import sys

    s_arg = sys.argv[1] if len(sys.argv) > 1 else "20180101000000"
    e_arg = sys.argv[2] if len(sys.argv) > 2 else "20251231235959"
    fetch_jra_data(s_arg, e_arg)
