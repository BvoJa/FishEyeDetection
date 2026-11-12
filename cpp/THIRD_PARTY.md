# C++ dependencies

`dependencies.lock.json` pins download URLs and SHA-256 checksums. Downloaded headers
and reference source stay in ignored `third_party/cpp/`. Build outputs stay in
ignored `build/cpp/`. License notices are retained in `cpp/licenses/`; distribute
them with a redistributed executable and the corresponding runtime library.

| Dependency | Version | Use | License |
|---|---|---|---|
| [ONNX Runtime](https://github.com/microsoft/onnxruntime/tree/v1.31.0) | 1.31.0 | C++ headers; CPU library from the installed Python wheel | MIT |
| [nlohmann/json](https://github.com/nlohmann/json/tree/v3.11.3) | 3.11.3 | Configuration and JSON output | MIT |
| [stb_image](https://github.com/nothings/stb/blob/f58f558c120e9b32c217290b80bad1a0729fbb2c/stb_image.h) | pinned commit | Native image decoding | MIT/public domain |
| [PicoSHA2](https://github.com/okdshin/PicoSHA2/tree/v1.0.1) | 1.0.1 | Model checksum validation | MIT |
| [Pillow](https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c) | 12.3.0 reference | Fixed-point bilinear coefficient/rounding compatibility | HPND |

The C++ resize implementation follows Pillow's two-pass, normalized, 22-bit
fixed-point bilinear algorithm. This preserves the preprocessing contract of the
existing Python runtime, including downsampling filter support and intermediate
8-bit rounding. The Pillow license is reproduced in `licenses/pillow.txt`.

The executable links the exact same `libonnxruntime.so.1.31.0` as Python. It does
not invoke Python during inference. Re-run the build helper after moving/replacing
the virtual environment, since the build uses library symlinks and an RPATH to
`third_party/cpp/lib/`. A standalone deployment must package the library, executable,
model/metadata pair and license notices together.
