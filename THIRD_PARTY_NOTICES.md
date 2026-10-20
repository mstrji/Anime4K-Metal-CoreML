# Third-party notices

## Anime4K

The shaders in `Sources/Anime4KUpscaler/Resources/Anime4K/*.glsl`, and the Core ML models converted
from their weights (`Anime4K_A_S.mlmodelc`, `Anime4K_C_S.mlmodelc`), come from
[bloc97/Anime4K](https://github.com/bloc97/Anime4K) (commit `7684e958`).

Copyright (c) 2019-2021 bloc97. MIT License — full text in
`Sources/Anime4KUpscaler/Resources/Anime4K/LICENSE-Anime4K.txt`.

## Anime4KMetal

The approach of translating mpv `//!HOOK` shaders into Metal compute kernels (parameters passed into
every function, mpv's texture macros spelled as Metal `#define`s) follows
[imxieyi/Anime4KMetal](https://github.com/imxieyi/Anime4KMetal) by Yi Xie, Copyright 2021,
licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
`MPVHookShader.swift` is a rewrite, not a copy of that code.
