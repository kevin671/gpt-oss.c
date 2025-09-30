#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>

/* Transformer */

typedef struct
{
    int n_layers;
    int n_experts;
    int n_experts_active;
    int vocab_size;
    int dim;
    int hidden_dim;
    int swiglu_limit;
    int head_dim;
    int n_heads;
    int n_kv_heads;
    int sliding_window;
    int init_context_length;
    int rope_theta;
    int rope_scaling_factor;
    int rope_ntk_alpha;
    int rope_ntk_beta;
} ModelConfig;

typedef struct
{
    int dim;        // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers;   // number of layers
    int n_heads;    // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len;    // max sequence length
} Config;

typedef struct
{
    int8_t *q;
    float *s;
} QuantizedTensor;

typedef struct
{
    QuantizedTensor *q_tokens; // (vocab_size, dim)
} TransformerWeights;

/* TODO: このモデルの定義は後で見直す  */
typedef struct
{
    // token embedding table
    QuantizedTensor *q_tokens;    // (vocab_size, dim)
    float *token_embedding_table; // same, but dequantized

    // weights for rmsnorms
    float *rms_att_weight; // (layer, dim) rmsnorm weights
    float *rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float *rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct
{
    // current wave of activations
    float *x;           // activation at current time stamp (dim,)
    float *xb;          // same, but inside a residual branch (dim,)
    float *xb2;         // an additional buffer just for convenience (dim,)
    float *hb;          // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2;         // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q;           // query (dim,)
    float *k;           // key (dim,)
    float *v;           // value (dim,)
    float *att;         // buffer for scores/attention values (n_heads, seq_len)
    float *logits;      // output logits
    // kv cache
    float *key_cache;   // (layer, seq_len, dim)
    float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct
{
    Config config;              // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state;             // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd;      // file descriptor for memory mapping
    float *data; // memory mapped data pointer
    // ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

/* Network Forward Pass */

void rmsnorm(float *o, float *x, float *weight, int size)
{
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++)
    {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++)
    {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float *x, int size)
{
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}

void matmul(float *xout, float *x, float *w, int n, int d)
{
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
#pragma omp parallel for private(i)
    for (i = 0; i < d; i++)
    {
        float val = 0.0f;
        for (int j = 0; j < n; j++)
        {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

float *forward(Transformer *transformer, int token, int pos)
{
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    float *x = s->x;
    memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(*x));

    for (int l = 0; l < p->n_layers; l++)
    {
        // prenorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        int loff = l * p->seq_len * kv_dim;
        
    }
}

/* Byte Pair Encoding (BPE) Tokenizer */

typedef struct
{
    char **vocab;
    int vocab_size;
    float *vocab_scores;
    unsigned int max_token_length;
    unsigned char byte_pieces[256 * 2];
    char **special_tokens;
} Tokenizer;

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size)
{
    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));

    for (int i = 0; i < 256; i++)
    {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    FILE *f = fopen(tokenizer_path, "r");
    fread(&t->max_token_length, sizeof(int), 1, f);
    int len;
    for (int i = 0; i < vocab_size; i++)
    {
        fread(&t->vocab_scores[i], sizeof(float), 1, f);
        fread(&len, sizeof(int), 1, f);
        t->vocab[i] = (char *)malloc((len + 1) * sizeof(char));
        fread(t->vocab[i], len, 1, f);
        t->vocab[i][len] = '\0';
    }
    fclose(f);
}

void free_tokenizer(Tokenizer *t)
{
    for (int i = 0; i < t->vocab_size; i++)
    {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
}

char *decode(Tokenizer *t, int prev_token, int token)
{
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ')
    {
        piece++;
    }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
    {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

/* generate */

int main(int argc, char **argv) // ./a.out checkpoint.bin 201088 72 329 50256 123 ...
{

    char *tokenizer_path = "tokenizer.bin"; // argv[1];
    int vocab_size = 201088;                // atoi(argv[2]);

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, vocab_size);

    int num_tokens = argc - 3;
    int *tokens = (int *)malloc(num_tokens * sizeof(int));
    for (int i = 0; i < num_tokens; i++)
    {
        tokens[i] = atoi(argv[i + 3]);
    }

    int prev_token = -1;
    for (int i = 0; i < num_tokens; i++)
    {
        char *decoded_piece = decode(&tokenizer, prev_token, tokens[i]);
        printf("%s", decoded_piece);
        prev_token = tokens[i];
    }
    printf("\n");

    free(tokens);
    free_tokenizer(&tokenizer);
    return 0;
}