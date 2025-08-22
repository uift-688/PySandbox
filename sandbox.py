import asyncio
import ast
from types import ModuleType
from typing import Callable, Union, Any, Iterable, Type
from asteval import Interpreter
from fs.osfs import OSFS
import fs
from pathlib import Path
from types import ModuleType
import os
import shutil
from fs.path import combine, normpath
from dataclasses import dataclass
from gc import collect
import psutil
from functools import lru_cache
from collections import deque
import builtins

_socket_registry = {}

pid = os.getpid()
process = psutil.Process(pid)

class VirtualSocket:
    def __init__(self, family=0, type=0, proto=0):
        self.family = family
        self.type = type
        self.proto = proto
        self.address = None
        self.recv_queue = deque()
        self.connected = False
        self.closed = False
        self.is_server = False
        self.client_sockets = []
        self.peer = None

    def _check_closed(self):
        if self.closed:
            raise RuntimeError("Socket is closed")

    def bind(self, address):
        self._check_closed()
        if address in _socket_registry:
            raise OSError("Address already in use")
        self.address = address
        _socket_registry[address] = self

    def listen(self, backlog=5):
        self._check_closed()
        self.is_server = True
        self.backlog = backlog

    def accept(self):
        self._check_closed()
        if not self.is_server:
            raise RuntimeError("Socket is not a server")
        if not self.client_sockets:
            return None, None
        client = self.client_sockets.pop(0)
        return client, client.address

    def connect(self, address):
        self._check_closed()
        if address not in _socket_registry:
            raise OSError("No such address")
        server = _socket_registry[address]
        if not server.is_server:
            raise RuntimeError("Target is not a listening socket")
        self.peer = server
        server.client_sockets.append(self)
        self.connected = True

    def send(self, data, flags=0):
        self._check_closed()
        if not self.connected or not self.peer:
            raise RuntimeError("Socket not connected")
        self.peer.recv_queue.append(data)
        return len(data)

    def sendall(self, data, flags=0):
        self._check_closed()
        if not self.connected or not self.peer:
            raise RuntimeError("Socket not connected")
        self.peer.recv_queue.append(data)

    def recv(self, bufsize, flags=0):
        self._check_closed()
        if not self.connected and not self.is_server:
            raise RuntimeError("Socket not connected")
        if not self.recv_queue:
            return None
        data = self.recv_queue.popleft()
        return data[:bufsize]

    def close(self):
        if self.closed:
            return
        if self.address and self.is_server:
            _socket_registry.pop(self.address, None)
        self.closed = True
        self.connected = False
        self.peer = None
        self.recv_queue.clear()
        self.client_sockets.clear()

import socket

class VirtualSocketModule(ModuleType):
    socket = VirtualSocket

for name, value in vars(socket).items():
    if name.isupper():
        if not callable(value):
            setattr(VirtualSocketModule, name, value)

@dataclass(frozen=True, init=False)
class Builtins:
    def len(self, obj: Iterable) -> int:
        return len(obj)

    def sum(self, obj: Iterable) -> float:
        return sum(obj)

    def max(self, obj: Iterable) -> Any:
        return max(obj)

    def min(self, obj: Iterable) -> Any:
        return min(obj)

    def sorted(self, obj: Iterable, reverse: bool = False) -> list:
        return sorted(obj, reverse=reverse)

    def reversed(self, obj: Iterable) -> list:
        return list(reversed(obj))

    def enumerate(self, obj: Iterable, start: int = 0) -> list[tuple[int, Any]]:
        return list(enumerate(obj, start=start))

    def zip(self, *iterables: Iterable) -> list[tuple]:
        return list(zip(*iterables))

    def all(self, obj: Iterable) -> bool:
        return all(obj)

    def any(self, obj: Iterable) -> bool:
        return any(obj)

    def isinstance(self, obj: Any, cls: Type) -> bool:
        return isinstance(obj, cls)

    def issubclass(self, cls: Type, classinfo: Type) -> bool:
        return issubclass(cls, classinfo)

    def map(self, func: Callable, iterable: Iterable) -> list:
        return list(map(func, iterable))

    def filter(self, func: Callable, iterable: Iterable) -> list:
        return list(filter(func, iterable))

    def int(self, obj: Any) -> int:
        return int(obj)

    def float(self, obj: Any) -> float:
        return float(obj)

    def str(self, obj: Any) -> str:
        return str(obj)

    def bool(self, obj: Any) -> bool:
        return bool(obj)

    def complex(self, obj: Any) -> complex:
        return complex(obj)

    # 数値系
    def abs(self, obj: Any) -> Any:
        return abs(obj)

    def round(self, obj: Any, ndigits = None) -> Any:
        return round(obj, ndigits) if ndigits is not None else round(obj)

    def divmod(self, a: Any, b: Any) -> tuple:
        return divmod(a, b)

    def pow(self, a: Any, b: Any) -> Any:
        return pow(a, b)

class OverlayScope:
    def __init__(self, fields: list[dict]):
        self.fields = tuple(fields)
    def __getitem__(self, name):
        for scope in reversed(self.fields):
            if name in scope:
                return scope[name]
            else:
                continue
    def __setitem__(self, name, value):
        for scope in reversed(self.fields):
            if name in scope:
                scope[name] = value
            else:
                continue
    def __delitem__(self, name):
        for scope in reversed(self.fields):
            if name in scope:
                del scope[name]
            else:
                continue

class AsyncCachedModuleInterpreter(Interpreter):
    def __init__(self, path: Union[str, Path], modules: dict[str, Callable[[], ModuleType]] = None, max_memory: int = None, deceleration_time: Union[float, None] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modules = {**(modules or {}), "socket": VirtualSocketModule}
        self._module_cache: dict[str, ModuleType] = {}
        self.globals = {}
        self.filesystem = self.root_filesystem = OSFS(path, True)
        self.shutil = ModuleType("shutil")
        self.modules["os"] = self._module_os
        self.modules["shutil"] = self._module_shutil
        self.symtable["open"] = self._default_open
        self.working_dir = "/"
        builtins_scope = vars(Builtins).copy()
        self.scope = OverlayScope([builtins_scope, self.globals, self.symtable])
        self.get_memory = lru_cache(2)(process.memory_info)
        self.max_memory = max_memory
        self._o_exec = None
        self._o_eval = None
        self._o_open = None
        self._o_import = None
        self._o_sub = object.__subclasses__
        self._o_base = type.__base__
        self.locals = self.symtable
        self.performance_time = deceleration_time

    def _default_chdir(self, path):
        self.working_dir = normpath(combine(self.working_dir, path))
        self.filesystem = self.root_filesystem.opendir(self.working_dir)

    def _module_os(self):
        self.os = ModuleType("os")
        for key, value in vars(fs).items():
            if key.startswith("__") and key.endswith("__"):
                continue
            if hasattr(os, key):
                setattr(self.os, key, value)
        for key, value in vars(self.filesystem).items():
            if key.startswith("__") and key.endswith("__"):
                continue
            if hasattr(os, key):
                setattr(self.os, key, value)
        self.os.chdir = self._default_chdir
        return self.os

    def _module_shutil(self):
        for key, value in vars(fs).items():
            if key.startswith("__") and key.endswith("__"):
                continue
            if hasattr(shutil, key):
                setattr(self.shutil, key, value)
        for key, value in vars(self.filesystem).items():
            if key.startswith("__") and key.endswith("__"):
                continue
            if hasattr(shutil, key):
                setattr(self.shutil, key, value)
        return self.shutil

    def _get_module(self, name: str) -> ModuleType:
        if name in self._module_cache:
            return self._module_cache[name]
        if name not in self.modules:
            raise ImportError(f"Module '{name}' is not allowed")
        module = self.modules[name]()
        self._module_cache[name] = module
        return module

    def on_import(self, node: ast.Import):
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            self.symtable[asname] = self._get_module(name)

    def on_importfrom(self, node: ast.ImportFrom):
        module_name = node.module
        module = self._get_module(module_name)
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            self.symtable[asname] = getattr(module, name)

    def _default_open(self, *args, **kwargs):
        return self.filesystem.open(*args, **kwargs)

    async def aeval(self, expr: str):
        tree = self.parse(expr)
        return await self._avisit(tree.body[0])

    async def _avisit(self, node):
        try:
            if self.performance_time is not None:
                await asyncio.sleep(self.performance_time / 1000)
            if isinstance(node, ast.Await):
                val = await self._avisit(node.value)
                return await val
            elif isinstance(node, ast.Expr):
                return await self._avisit(node.value)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Name):
                return self.scope[node.id]
            elif isinstance(node, ast.Assign):
                value = await self._avisit(node.value)
                for name in node.targets:
                    if isinstance(name, ast.Name):
                        self.symtable[name.id] = value
                    elif isinstance(name, ast.Tuple):
                        for i, elt in enumerate(name.elts):
                            if isinstance(elt, ast.Name):
                                self.symtable[elt.id] = value[i]
            elif isinstance(node, ast.Call):
                func = await self._avisit(node.func)
                args = [await self._avisit(a) for a in node.args]
                kwargs = {kw.arg: await self._avisit(kw.value) for kw in node.keywords}
                return func(*args, **kwargs)
            elif isinstance(node, ast.Attribute):
                value = await self._avisit(node.value)
                return getattr(value, node.attr)
            else:
                return self._eval_node(node)
        finally:
            if self.max_memory is not None:
                memory = self.get_memory().rss
                if self.max_memory < memory:
                    collect()
                    if process.memory_info().rss < memory:
                        raise MemoryError
    def _eval_node(self, node):
        method = f"on_{node.__class__.__name__.lower()}"
        if hasattr(self, method):
            return getattr(self, method)(node)
        raise NotImplementedError(f"No _visit method for {type(node).__name__}")
    async def __aenter__(self):
        self._o_eval, self._o_exec, self._o_import, self._o_open = eval, exec, __import__, open
        builtins.eval = self.aeval
        builtins.exec = self.aeval
        builtins.__import__ = empty
        builtins.open = empty
        object.__subclasses__ = empty
        type.__base__ = property(empty)
        return self
    async def __aexit__(self, e, ev, tb):
        builtins.eval = self._o_eval
        builtins.exec = self._o_exec
        builtins.__import__ = self._o_import
        builtins.open = self._o_open
        object.__subclasses__ = self._o_sub
        type.__base__ = self._o_base

def empty():
    pass
