# Embedding model based on transformer

Ref: https://github.com/karpathy/build-nanogpt/blob/01be6b358941cb1c7561a56353423eba1cc7fe80/train_gpt2.py

```
poetry run python -m demo_embedding.training 
```

Colab: https://colab.research.google.com/drive/1IUrbHn6v5F7-t9pL19VegJSx-GjqBESp?usp=sharing

## Feature extractions

- transformers/models/distilbert/modeling_distilbert.py - `last_hidden_state`
- https://blog.min.io/feature-extraction-with-large-language-models-hugging-face-and-minio/

## Links

- https://www.youtube.com/watch?v=l8pRSuU81PU&ab_channel=AndrejKarpathy
- https://github.com/karpathy/build-nanogpt/blob/01be6b358941cb1c7561a56353423eba1cc7fe80/train_gpt2.py
- https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py
- https://github.com/dosuken123/demo_embedding
- [Sentence Transformer](https://www.sbert.net/docs/quickstart.html)