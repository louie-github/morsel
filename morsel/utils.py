#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

from typing import Any, Type, TypeVar


T = TypeVar("T")


def convert_and_warn(obj: Any, typ: Type[T], name: str = None) -> T:
    if isinstance(obj, typ):
        return obj
    else:
        name = f'"{name}"' if name is not None else "an object"
        warnings.warn(
            f'Implicitly converting {name} from type "{type(obj)}" to type "{typ}"'
        )
        return typ(obj)
