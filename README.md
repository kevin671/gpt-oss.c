## gpt-oss.c

- https://github.com/openai/gpt-oss
- https://github.com/karpathy/llama2.c

```
make runq
python export.py gptoss_20b.bin --version 0 --model_path path/to/gpt-oss/model/20B
./runq gptoss_20b.bin -t 0.0 -n 256 -i "One day, a little girl named Lily found a needle in her room"
```
