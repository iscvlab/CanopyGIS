import importlib.util
import os
import subprocess
import sys

if __name__ == "__main__":
    if importlib.util.find_spec("samgeo") is None or importlib.util.find_spec("deepforest") is None:
        candidate = os.path.join(os.path.expanduser("~"), "Miniconda3", "envs", "nerfstudio", "python.exe")
        if os.path.isfile(candidate) and os.path.normcase(sys.executable) != os.path.normcase(candidate):
            raise SystemExit(subprocess.call([candidate, __file__, *sys.argv[1:]]))

    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="127.0.0.1", port=port)
