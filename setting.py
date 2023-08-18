from __future__ import unicode_literals

import functools
import hashlib
import json
import os
import time
import uuid
from datetime import date, datetime
from decimal import Decimal
from json import JSONEncoder

cretKey = os.getenv("SECRET_KEY", "iUe2sdaafgsdaghdsfhgsdffasfB")

dbUrl = os.getenv(
    "DB_URL",
    "mysql://tudb_ai:QP4NbGDyb6vc@bjtudb.rwlb.rds.aliyuncs.com:3306/ai_painting?charset=utf8mb4&maxsize=20")


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]

    def copy(self):
        n = Map(self.__dict__.copy())
        return n


def group_list(l, size):
    lc = l.copy()
    ret = []
    if len(l) > size:
        while len(lc) >= size:
            ret.append(lc[:size])
            lc = lc[size:]
        if len(lc) > 0:
            ret.append(lc)
        return ret
    else:
        return [l]


class MyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, date):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, Decimal):
            return format(obj, 'f')
        else:
            return super(MyEncoder, self).default(obj)


# class MyDecoder(JSONDecoder):
def set_object_hook(obj):
    if isinstance(obj, dict):
        return Map(obj)
    return obj


def loads(str):
    return json.loads(str, object_hook=set_object_hook, strict=False)


def dumps(obj):
    return json.dumps(obj, cls=MyEncoder, ensure_ascii=False)


def md5(content):
    return hashlib.md5(content.encode(encoding='UTF-8')).hexdigest()


def uuid_str(): return str(uuid.uuid4()).replace("-", "")


def time_cache(max_age, maxsize=128, typed=False):
    """Least-recently-used cache decorator with time-based cache invalidation.

    Args:
        max_age: Time to live for cached results (in seconds).
        maxsize: Maximum cache size (see `functools.lru_cache`).
        typed: Cache on distinct input types (see `functools.lru_cache`).
    """

    def _decorator(fn):
        @functools.lru_cache(maxsize=maxsize, typed=typed)
        def _new(*args, __time_salt, **kwargs):
            return fn(*args, **kwargs)

        @functools.wraps(fn)
        def _wrapped(*args, **kwargs):
            return _new(*args, **kwargs, __time_salt=int(time.time() / max_age))

        return _wrapped

    return _decorator
