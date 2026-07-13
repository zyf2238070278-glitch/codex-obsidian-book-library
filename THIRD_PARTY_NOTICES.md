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

模型权重随 all-in-one ZIP 分发，安装后从本机 `data/models` 加载。模型名称、固定 revision 和来源链接用于标识所分发的确切权重；上游模型卡中的说明和归属仍然适用。
