# Third-Party Notices

This distribution directly includes the following third-party components. The project license in `LICENSE` does not replace their licenses.

## uv 0.11.26

- Component: `uv 0.11.26` macOS Apple Silicon executable
- Upstream: https://github.com/astral-sh/uv/tree/0.11.26
- License: Apache-2.0 OR MIT, at the recipient's option
- License texts: `third_party/uv/LICENSE-APACHE` and `third_party/uv/LICENSE-MIT`

The executable is placed at `bin/uv` in the all-in-one ZIP and is used to prepare the local Python environment.

## intfloat/multilingual-e5-small

- Component: `intfloat/multilingual-e5-small` semantic embedding model
- Revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`
- Upstream: https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3
- License declared by the upstream model card: MIT
- License text: `third_party/model/LICENSE-MIT`

The pinned model repository does not contain a separate LICENSE file.
`third_party/model/LICENSE-MIT` is a standard MIT license text supplied for
license reference based on the upstream model-card metadata; it is not
presented as a verbatim upstream license file.

模型权重随 all-in-one ZIP 分发，安装后从本机 `data/models` 加载。模型名称、固定 revision 和来源链接用于标识所分发的确切权重；上游模型卡中的说明和归属仍然适用。

## @arcships/light-ocr 0.3.0

- Components: `@arcships/light-ocr`, `@arcships/light-ocr-darwin-arm64`, and `@arcships/light-ocr-model-ppocrv6-small` 0.3.0
- Upstream: https://github.com/arcships/light-ocr/tree/v0.3.0
- License: Apache-2.0
- License and notice files: installed with each package beneath `node_modules/@arcships/`

Light OCR is the third local OCR fallback on Apple Silicon macOS. Its native runtime and PP-OCRv6 Small model remain on the local machine and do not download models while recognizing pages.
