# Test data

`ggml-vocab-qwen2.gguf`, `.inp` and `.out` are copied unchanged from the
llama.cpp repository (`models/`, MIT licence, https://github.com/ggml-org/llama.cpp).
The GGUF holds only the Qwen2 vocabulary (no weights); `.inp` holds test strings
separated by `\n__ggml_vocab_test__\n` and `.out` the token ids llama.cpp produces
for each. `tests/test_tok.c` checks our tokenizer against them.
