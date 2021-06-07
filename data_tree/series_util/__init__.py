from data_tree._series import Series, ListLikeSeries
from data_tree.ops.cache import CachedSeries


def make_series_sharable(s: Series, manager):
    return list_to_managed_list(
        remove_cache(s),
        manager
    )


def remove_cache(s: Series):
    if isinstance(s, CachedSeries):
        return remove_cache(s.src)
    else:
        return s.clone([remove_cache(p) for p in s.parents])


def list_to_managed_list(s: Series, manager):
    if isinstance(s, ListLikeSeries):
        managed = manager.list(s.data)
        return ListLikeSeries(managed)
    else:
        return s.clone([list_to_managed_list(p, manager) for p in s.parents])
