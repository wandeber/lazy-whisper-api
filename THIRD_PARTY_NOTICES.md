# Third-party notices and attribution

The source code authored for lazy-whisper-api is licensed under the MIT
License in [LICENSE](LICENSE). Model weights, converted weights, Python
packages, native libraries, and executables retain their upstream licenses.
The MIT license for this repository does not relicense those materials.

## Scope

The Git repository does not contain the configured ASR or diarization model
weights, the Silero ONNX asset, third-party Python wheels, or an FFmpeg
executable. Setup and package installation obtain those materials separately.
The committed `bin/ffmpeg` file is only a launcher for the executable installed
by `imageio-ffmpeg`.

This document records the origin, attribution, license, and modification status
of the models and principal inference/media components used by the default
configuration. A distributor that bundles any of them must also preserve the
complete license files, copyright notices, and any upstream `NOTICE` files
shipped with the exact artifact. Links in this document are not substitutes for
those files when a license requires recipients to receive a copy.

## Model attribution

### Qwen3-ASR and Qwen3-ForcedAligner

- **Materials:** `Qwen/Qwen3-ASR-0.6B`, `Qwen/Qwen3-ASR-1.7B`, and
  `Qwen/Qwen3-ForcedAligner-0.6B`.
- **Creator and copyright notice:** Qwen Team; the upstream Qwen3-ASR software
  license carries `Copyright 2026 Alibaba Cloud`.
- **Sources:** [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B),
  [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B),
  [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B),
  and the [Qwen3-ASR source repository](https://github.com/QwenLM/Qwen3-ASR).
- **License:** [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- **Changes:** lazy-whisper-api does not modify or redistribute the upstream
  weights. `qwen-1.7b-edit-max` is a local API alias for the unmodified
  `Qwen3-ASR-1.7B` model. Forced alignment, Silero/energy boundary fusion, and
  response construction are project-side processing and are not presented as
  modifications to the Qwen models.

The upstream model card requests the following research citation:

> Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang,
> Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, Jin Xu, Jingren Zhou, and
> Junyang Lin. “Qwen3-ASR Technical Report.” arXiv:2601.21337 (2026).

If the models or `qwen-asr` software are redistributed, recipients must receive
the Apache-2.0 license, modified files must be marked when applicable, and all
applicable upstream attribution and `NOTICE` content must be retained.

### Whisper converted weights

- **Materials:** `Systran/faster-whisper-large-v3` and
  `dropbox-dash/faster-whisper-large-v3-turbo`.
- **Creators:** the original Whisper models are by OpenAI; the CTranslate2
  conversions are published by SYSTRAN and Dropbox Dash respectively.
- **Sources:**
  [SYSTRAN large-v3 conversion](https://huggingface.co/Systran/faster-whisper-large-v3),
  [Dropbox Dash large-v3-turbo conversion](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo),
  and [OpenAI Whisper](https://github.com/openai/whisper).
- **License and copyright:** MIT; the original Whisper license carries
  `Copyright (c) 2022 OpenAI`.
- **Changes:** lazy-whisper-api downloads/loads the configured converted model
  and does not modify or redistribute its weights.

### pyannote speaker diarization (CC BY 4.0 attribution)

- **Material/title:** `pyannote/speaker-diarization-community-1`.
- **Creator/publisher:** pyannote.
- **Source:** [official model repository and model card](https://huggingface.co/pyannote/speaker-diarization-community-1).
- **License:** [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- **Changes:** lazy-whisper-api does not modify or redistribute the model
  weights. It loads a locally downloaded copy and combines the model's speaker
  turns with transcription timestamps. No endorsement by pyannote is implied.

The source link, creator, license link, and change statement above are the
attribution supplied for CC BY 4.0. The license's
[warranty disclaimer and limitation of liability](https://creativecommons.org/licenses/by/4.0/legalcode#s5)
apply. The Hugging Face repository is gated: each operator must accept its
current access conditions before downloading the model. Those access
conditions are not removed or replaced by this project.

The model card requests citations for its three principal components:

1. Alexis Plaquet and Hervé Bredin. “Powerset multi-class cross entropy loss
   for neural speaker diarization.” Proc. INTERSPEECH 2023.
2. Hongji Wang, Chengdong Liang, Shuai Wang, Zhengyang Chen, Binbin Zhang,
   Xu Xiang, Yanlei Deng, and Yanmin Qian. “WeSpeaker: A research and production
   oriented speaker embedding learning toolkit.” ICASSP 2023, pages 1–5.
3. Federico Landini, Ján Profant, Mireia Diez, and Lukáš Burget. “Bayesian HMM
   clustering of x-vector sequences (VBx) in speaker diarization: theory,
   implementation and analysis on standard tasks.” Computer Speech & Language
   (2022).

## Principal runtime and signal-processing components

These packages/assets are installed dependencies rather than source copied into
this repository. The versions below are the repository's pinned/default
versions at the time of this notice.

| Component | Attribution | License | Upstream |
| --- | --- | --- | --- |
| `qwen-asr` 0.0.6 | Copyright 2026 Alibaba Cloud | Apache-2.0 | [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) |
| `mlx-qwen3-asr` 0.3.5 | Copyright 2025 dmoon | Apache-2.0 | [mlx-qwen3-asr](https://github.com/moona3k/mlx-qwen3-asr) |
| `faster-whisper` 1.2.1 | Copyright (c) 2023 SYSTRAN | MIT | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| Silero VAD asset | Copyright (c) 2020-present Silero Team | MIT | [silero-vad](https://github.com/snakers4/silero-vad) |
| ONNX Runtime | Copyright (c) Microsoft Corporation | MIT | [onnxruntime](https://github.com/microsoft/onnxruntime) |
| NumPy 2.4.6 | Copyright (c) 2005-2025, NumPy Developers | BSD-3-Clause plus bundled third-party licenses | [NumPy](https://github.com/numpy/numpy) |
| `pyannote.audio` 4.0.7 | Copyright (c) 2020 CNRS | MIT | [pyannote.audio](https://github.com/pyannote/pyannote-audio) |
| PyAV 17.0.0 | Copyright retained by original committers | BSD-3-Clause | [PyAV](https://github.com/PyAV-Org/PyAV) |
| `imageio-ffmpeg` 0.6.0 Python package | Copyright (c) 2019-2025, imageio | BSD-2-Clause | [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) |

The NumPy 2.4.6 package metadata declares
`BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`. A binary redistribution
must preserve the complete license collection from NumPy's installed
`*.dist-info/licenses` directory, not only the top-level BSD-3-Clause notice.
The same rule applies to other packages that ship a license directory or
multiple component notices.

Silero VAD is loaded from Faster Whisper's installed local ONNX asset. The
asset is not modified. The edit-max profile adds original hysteresis and local
energy-edge processing around its output; it does not claim that processing as
part of Silero VAD.

## FFmpeg and binary redistribution

`imageio-ffmpeg` is BSD-2-Clause, but its platform wheels include a separately
licensed FFmpeg executable. FFmpeg is primarily LGPL-2.1-or-later; an
executable built with `--enable-gpl` is GPL-2.0-or-later. The macOS arm64
FFmpeg 7.1 executable installed by the currently pinned wheel reports
`--enable-gpl` in `ffmpeg -version`, so that executable must be treated as
GPL-2.0-or-later for redistribution. Other platforms/builds must be checked
from their own build configuration instead of assuming the same result.

This repository does not commit or redistribute that executable. Anyone who
ships a virtual environment, container, desktop bundle, or other artifact that
contains it must comply with the applicable FFmpeg license, include its license
and copyright notices, and provide the corresponding source or source offer as
required by that license. PyAV wheels may also contain FFmpeg shared libraries;
their exact wheel license files and native-library notices must accompany a
redistribution.

Commercial use is permitted by the licenses listed here, but GPL/LGPL binary
redistribution has copyleft/source-delivery obligations. That is a distribution
condition on the bundled FFmpeg artifact and does not change the MIT license of
the lazy-whisper-api source code.

## Locally supplied artifacts and dependency notices

The configured `distil-multi4` source is a local path and its artifact is not
distributed in this repository. Operators may also point configuration at
other local or Hugging Face models. Their provenance and license must be checked
separately; a path, repository ID, or successful download is not a license
grant.

Python dependencies and their transitive native libraries can vary by platform
and selected runtime. When redistributing an installed or frozen environment,
retain every `LICENSE*`, `COPYING*`, `NOTICE*`, and `*.dist-info/licenses`
entry supplied by those exact packages and review the resulting dependency set.
This notice is an attribution record for the repository configuration, not a
substitute for an artifact-specific software bill of materials or legal advice.
