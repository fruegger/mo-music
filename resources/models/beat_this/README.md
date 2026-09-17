# Beat This! ONNX export

Regenerates the `.rten` model `audan-beats`' `onnx-beats` feature loads, from
the upstream [CPJKU/beat_this](https://github.com/CPJKU/beat_this) PyTorch
checkpoint (`final0`, MIT-licensed code and weights).

## Steps

```sh
python -m venv .venv
# Windows: .venv\Scripts\python.exe   |   Linux/macOS: .venv/bin/python
PY=.venv/bin/python

$PY -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
$PY -m pip install numpy einops rotary-embedding-torch soxr onnx onnxruntime rten-convert beat-this

$PY export.py beat_this_final0.onnx
```

This loads the `final0` checkpoint, exports it to ONNX with a fixed
`(1, 1500, 128)` input shape (batch=1, 1500 frames = 30s at 50fps, 128 mel
bands -- `audan-beats`' chunking logic always feeds exactly this shape, see
`crates/audan-beats/src/model.rs`), and verifies the export matches the
PyTorch reference numerically (expect agreement around `1e-5`-`1e-6`).

Then convert to rten's format:

```sh
$PY -m rten_convert --no-infer-shapes beat_this_final0.onnx beat_this_final0.rten
```

**`--no-infer-shapes` is required on Windows**: `rten-convert`'s default
shape-inference step uses `tempfile.NamedTemporaryFile` and then reopens that
same path from another call while the handle is still open, which Windows'
file locking rejects (`PermissionError: ... Access is denied`). This doesn't
affect the correctness of the converted model -- shape inference is an
optional graph-optimization aid, not required for the model to load or run
correctly. On Linux/macOS `--infer-shapes` (the default) works fine and can
be left on.

The resulting `beat_this_final0.rten` is what gets hosted (see
`resources/models/registry.json`'s `beat_this` entry) and loaded by
`audan_beats::model::OnnxBackend`.
