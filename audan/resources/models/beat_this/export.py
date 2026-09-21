"""Export the "Beat This!" (CPJKU/beat_this, ISMIR 2024) beat-tracking
checkpoint to ONNX, then verify it numerically against the PyTorch reference.

This regenerates the artifact audan-beats' `onnx-beats` feature loads (after
a separate `rten-convert` step -- see this directory's README.md). It is not
run as part of the normal Rust build; it is a one-off (or re-run-when-needed)
tool for producing that artifact from the upstream checkpoint.

Usage (from an environment with the packages listed in README.md installed):
    python export.py [output.onnx]
"""

import sys
import traceback

import torch
from beat_this.inference import load_model

CHUNK = 1500
NMELS = 128
CHECKPOINT = "final0"


class DictToStackedOutput(torch.nn.Module):
    """torch.onnx.export doesn't need a dict-valued graph output; stack
    beat/downbeat into one tensor of shape (2, batch, time) instead."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return torch.stack([out["beat"], out["downbeat"]], dim=0)


def main():
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else "beat_this_final0.onnx"

    print("torch:", torch.__version__)
    print(f"loading checkpoint '{CHECKPOINT}'...")
    model = load_model(CHECKPOINT, device="cpu")
    model.eval()
    print("loaded OK, params:", sum(p.numel() for p in model.parameters()))

    torch.manual_seed(0)
    dummy = torch.randn(1, CHUNK, NMELS, dtype=torch.float32)

    with torch.inference_mode():
        ref = model(dummy)
    ref_beat = ref["beat"].detach().clone()
    ref_downbeat = ref["downbeat"].detach().clone()

    wrapped = DictToStackedOutput(model)

    print("exporting to ONNX (legacy exporter, opset 17)...")
    try:
        torch.onnx.export(
            wrapped,
            (dummy,),
            onnx_path,
            input_names=["spect"],
            output_names=["beat_downbeat"],
            opset_version=17,
            dynamo=False,
        )
    except Exception:
        print("export FAILED:")
        traceback.print_exc()
        sys.exit(1)
    print("exported ->", onnx_path)

    print("checking numeric parity against onnxruntime...")
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    onnx_out = sess.run([out_name], {in_name: dummy.numpy()})[0]

    beat_diff = np.max(np.abs(onnx_out[0] - ref_beat.numpy()))
    downbeat_diff = np.max(np.abs(onnx_out[1] - ref_downbeat.numpy()))
    print("max abs diff beat:", beat_diff)
    print("max abs diff downbeat:", downbeat_diff)

    tol = 1e-3
    if beat_diff < tol and downbeat_diff < tol:
        print(f"PARITY OK (within {tol})")
    else:
        print(f"PARITY FAILED (tolerance {tol})")
        sys.exit(1)


if __name__ == "__main__":
    main()
