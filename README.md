# PySandbox

A secure and flexible Python execution sandbox with AST-level control, virtual filesystem, and memory/network constraints.

## Features

* **AST-Level Execution:** Execute Python code directly at the AST level.
* **AST Node-Level Throttling:** Slow down execution on a per-node basis to prevent abuse.
* **Wrapped OS/Shutil:** Safe wrappers for `os` and `shutil`.
* **Virtual Filesystem:** Abstracted virtual filesystem using [pyfilesystem2](https://github.com/PyFilesystem/pyfilesystem2).
* **Virtual Socket Stack:** Simulate networking in a secure environment.
* **Memory Limitation:** Enforce memory usage limits (JVM-style).
* **Flexible Imports:** Granular control over importable modules.
* **Security Improvements:** Mitigates some known Python vulnerabilities.
* **MIT License:** Permissive open-source license.

## Dependencies

The following Python modules are required:

```text
asteval
fs
fs.osfs
psutil
```

> These are not installed via pip for this project; please ensure they are available in your Python environment.

## Usage

```python
import asyncio
from sandbox import AsyncCachedModuleInterpreter

async def main():
    sandbox = AsyncCachedModuleInterpreter(memory_limit=256*1024*1024)  # 256MB
    result = await sandbox.aeval("print('Hello from sandbox!')")
    print("Execution result:", result)

asyncio.run(main())
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
