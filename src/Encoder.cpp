#include "Encoder.hpp"
#include <random>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <omp.h>
json ConditioningEncoder::to_json() const {
    json j;
    j["dim"] = dim;
    j["vocab_size"] = vocab_size;
    // Ne pas inclure token_embeddings ici: c'est volumineux et doit être sauvegardé
    // via les formats binaires (SafeTensors/RawCheckpoint). On garde seulement des métadonnées.
    j["token_embeddings_size"] = token_embeddings.size();
    j["magik_prefix_count"] = magik_prefix_count;
    j["magik_prefix_weight"] = magik_prefix_weight;

    // Snapshot léger: embeddings des ids 0..(N-1) si disponibles.
    // Utile pour inspection/debug (les poids complets restent dans les tensors).
    if (!token_embeddings.empty() && dim > 0 && vocab_size > 0) {
        const int n = std::max(0, std::min(magik_prefix_count, vocab_size));
        if (n > 0) {
            json rows = json::array();
            for (int id = 0; id < n; ++id) {
                json row = json::array();
                const size_t base = static_cast<size_t>(id) * static_cast<size_t>(dim);
                for (int d = 0; d < dim; ++d) {
                    row.push_back(token_embeddings[base + static_cast<size_t>(d)]);
                }
                rows.push_back(row);
            }
            j["magik_token_embeddings_snapshot"] = rows;
        }
    }
    if (!seq_embedding.empty()) j["seq_embedding"] = seq_embedding;
    if (!mod_embedding.empty()) j["mod_embedding"] = mod_embedding;
    if (!mag_embedding.empty()) j["mag_embedding"] = mag_embedding;
    return j;
}

void ConditioningEncoder::from_json(const json &j) {
    if (j.contains("dim")) dim = j["dim"].get<int>();
    if (j.contains("vocab_size")) vocab_size = j["vocab_size"].get<int>();

    if (j.contains("magik_prefix_count")) magik_prefix_count = j["magik_prefix_count"].get<int>();
    if (j.contains("magik_prefix_weight")) magik_prefix_weight = j["magik_prefix_weight"].get<float>();

    if (j.contains("token_embeddings") && j["token_embeddings"].is_array()) {
        token_embeddings = j["token_embeddings"].get<std::vector<float>>();
    }

    if (j.contains("seq_embedding") && j["seq_embedding"].is_array()) {
        seq_embedding = j["seq_embedding"].get<std::vector<float>>();
    }
    if (j.contains("mod_embedding") && j["mod_embedding"].is_array()) {
        mod_embedding = j["mod_embedding"].get<std::vector<float>>();
    }
    if (j.contains("mag_embedding") && j["mag_embedding"].is_array()) {
        mag_embedding = j["mag_embedding"].get<std::vector<float>>();
    }

    // Cohérence minimale
    if (dim <= 0) throw std::runtime_error("ConditioningEncoder::from_json: invalid dim");
    if (vocab_size < 0) throw std::runtime_error("ConditioningEncoder::from_json: invalid vocab_size");
    if (magik_prefix_count < 0) magik_prefix_count = 0;
    if (!(magik_prefix_weight > 0.0f)) magik_prefix_weight = 1.0f;
    if (!token_embeddings.empty()) {
        size_t expected = static_cast<size_t>(std::max(0, vocab_size)) * static_cast<size_t>(dim);
        if (expected > 0 && token_embeddings.size() != expected) {
            // Tolérance: si vocab_size n'était pas fiable, on l'infère.
            if (token_embeddings.size() % static_cast<size_t>(dim) == 0) {
                vocab_size = static_cast<int>(token_embeddings.size() / static_cast<size_t>(dim));
            }
        }
    }
}
ConditioningEncoder::ConditioningEncoder(int d, int Size_Vo)
    : dim(d), vocab_size(0)
{
    if (dim <= 0) dim = 64;
    if (Size_Vo > 0) token_embeddings.reserve(static_cast<size_t>(Size_Vo) * static_cast<size_t>(dim));
    // leave vocab_size == 0 until ensureVocabSize is called
    std::cerr << "[encoder] create dim=" << dim << " reserve_vocab=" << Size_Vo << std::endl;
}

ConditioningEncoder::~ConditioningEncoder() = default;

void ConditioningEncoder::initRandom(uint64_t seed)
{
    std::mt19937 rng(static_cast<uint32_t>(seed ^ 0x9e3779b9u));
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
    if (vocab_size == 0) return;
    // Note: RNG n'est pas thread-safe, on garde séquentiel ici
    for (size_t i = 0; i < token_embeddings.size(); ++i) {
        token_embeddings[i] = dist(rng);
    }
}

void ConditioningEncoder::ensureSpecialEmbeddings(uint64_t seed)
{
    if (dim <= 0) throw std::runtime_error("ConditioningEncoder::ensureSpecialEmbeddings: invalid dim");
    std::mt19937 rng(static_cast<uint32_t>(seed ^ 0xA5A5A5A5u));
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);

    auto init_if_empty = [&](std::vector<float>& emb, uint32_t tweak) {
        if (emb.size() == static_cast<size_t>(dim)) return;
        emb.assign(static_cast<size_t>(dim), 0.0f);
        std::mt19937 lrng(rng());
        lrng.seed(static_cast<uint32_t>(seed ^ tweak));
        for (int d = 0; d < dim; ++d) {
            emb[static_cast<size_t>(d)] = dist(lrng);
        }
    };

    init_if_empty(seq_embedding, 0x13579BDFu);
    init_if_empty(mod_embedding, 0x2468ACE0u);
    init_if_empty(mag_embedding, 0x0F0F0F0Fu);
}

void ConditioningEncoder::ensureDim(int required_dim, uint64_t seed)
{
    if (required_dim <= 0) return;
    if (dim == required_dim) {
        ensureSpecialEmbeddings(seed);
        return;
    }

    // Ne pas détruire un encoder déjà chargé/entraîné.
    if (vocab_size > 0 || !token_embeddings.empty()) {
        throw std::runtime_error(
            "ConditioningEncoder::ensureDim: dim mismatch (have=" + std::to_string(dim) +
            ", need=" + std::to_string(required_dim) +
            "). Refusing to resize because token embeddings are already allocated."
        );
    }

    dim = required_dim;
    ensureSpecialEmbeddings(seed);
}

void ConditioningEncoder::sgdUpdateSpecialEmbeddings(const std::vector<float>& grad_text, float lr,
                                        bool update_seq, bool update_mod, bool update_mag)
{
    if (lr == 0.0f) return;
    if (grad_text.size() != static_cast<size_t>(dim)) return;

    const auto apply = [&](std::vector<float>& emb) {
        if (emb.size() != static_cast<size_t>(dim)) return;
        #pragma omp simd
        for (int d = 0; d < dim; ++d) {
            emb[static_cast<size_t>(d)] -= lr * grad_text[static_cast<size_t>(d)];
        }
    };

    if (update_seq) apply(seq_embedding);
    if (update_mod) apply(mod_embedding);
    if (update_mag) apply(mag_embedding);
}

void ConditioningEncoder::ensureVocabSize(size_t new_vocab_size, uint64_t seed)
{
    if (new_vocab_size <= static_cast<size_t>(vocab_size)) return;
    std::cerr << "[encoder] resize_vocab old=" << vocab_size << " new=" << new_vocab_size
              << " dim=" << dim << " seed=" << seed << std::endl;
    const size_t old   = static_cast<size_t>(vocab_size);
    const size_t dim_t = static_cast<size_t>(dim);
    token_embeddings.resize(new_vocab_size * dim_t);

    const size_t start = old * dim_t;
    const size_t n_new = token_embeddings.size() - start;

    // Initialisation parallèle : chaque thread utilise son propre RNG
    // pour éviter tout verrou et saturer les coeurs disponibles.
    #pragma omp parallel if(n_new > 65536)
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        // Graine distincte par thread pour garantir la reproductibilité.
        std::mt19937 rng(static_cast<uint32_t>(
            (seed ^ 0xC0FFEEu) ^ (static_cast<uint64_t>(tid + 1) * 0x9e3779b97f4a7c15ULL)));
        std::uniform_real_distribution<float> dist(-0.02f, 0.02f);

        const size_t chunk = (n_new + static_cast<size_t>(nth) - 1) / static_cast<size_t>(nth);
        const size_t lo    = start + std::min(static_cast<size_t>(tid)     * chunk, n_new);
        const size_t hi    = start + std::min(static_cast<size_t>(tid + 1) * chunk, n_new);
        for (size_t i = lo; i < hi; ++i) {
            token_embeddings[i] = dist(rng);
        }
    }

    vocab_size = static_cast<int>(new_vocab_size);
}

static inline uint32_t mix_u32(uint32_t x) {
    // Simple mix (xorshift + avalanching) for deterministic pseudo-random weights.
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

void ConditioningEncoder::setMagicFromToken(const MagicToken &mt)
{
    ensureSpecialEmbeddings();
    mag_embedding.assign(static_cast<size_t>(dim), 0.0f);

    const uint32_t base_seed = mix_u32(mt.seed ^ (mt.modality_mask * 0x9e3779b1u) ^ 0xC0FFEEu);

    // Projection 8-d -> dim avec poids pseudo-aléatoires déterministes.
    // Parallélisé : chaque dimension est indépendante.
    #pragma omp parallel for schedule(static) if(dim > 256)
    for (int d = 0; d < dim; ++d) {
        float acc = 0.0f;
        for (int i = 0; i < 8; ++i) {
            uint32_t h = mix_u32(base_seed
                ^ (static_cast<uint32_t>(d + 1) * 0xA341316Cu)
                ^ (static_cast<uint32_t>(i + 1) * 0xC8013EA4u));
            float w = (static_cast<int>(h % 20001u) - 10000) / 100000.0f; // *= 0.1 / 10000
            acc += mt.embed[i] * w;
        }
        mag_embedding[static_cast<size_t>(d)] = acc;
    }

    // Normalisation L2.
    double ss = 0.0;
    #pragma omp simd reduction(+:ss)
    for (int i = 0; i < dim; ++i) {
        const double v = static_cast<double>(mag_embedding[static_cast<size_t>(i)]);
        ss += v * v;
    }
    const double n = std::sqrt(ss);
    if (n > 1e-12) {
        const float inv = static_cast<float>(1.0 / n);
        #pragma omp simd
        for (int i = 0; i < dim; ++i) mag_embedding[static_cast<size_t>(i)] *= inv;
    }
}

void ConditioningEncoder::setSeqEmbedding(const std::vector<float> &v)
{
    if (v.empty()) {
        seq_embedding.clear();
        return;
    }
    seq_embedding.assign(static_cast<size_t>(dim), 0.0f);
    size_t n = std::min<size_t>(v.size(), static_cast<size_t>(dim));
    std::copy_n(v.data(), n, seq_embedding.data());
}

void ConditioningEncoder::setModEmbedding(const std::vector<float> &v)
{
    if (v.empty()) {
        mod_embedding.clear();
        return;
    }
    mod_embedding.assign(static_cast<size_t>(dim), 0.0f);
    size_t n = std::min<size_t>(v.size(), static_cast<size_t>(dim));
    std::copy_n(v.data(), n, mod_embedding.data());
}

void ConditioningEncoder::setMagEmbedding(const std::vector<float> &v)
{
    if (v.empty()) {
        mag_embedding.clear();
        return;
    }
    mag_embedding.assign(static_cast<size_t>(dim), 0.0f);
    size_t n = std::min<size_t>(v.size(), static_cast<size_t>(dim));
    std::copy_n(v.data(), n, mag_embedding.data());
}

std::vector<float> ConditioningEncoder::encode(const std::vector<int> &tokens, uint32_t /*seed*/) const
{
    std::cerr << "[encoder] encode tokens=" << tokens.size() << " dim=" << dim << std::endl;
    std::vector<float> out(static_cast<size_t>(dim), 0.0f);
    encodeInto(out, tokens);
    return out;
}

void ConditioningEncoder::encodeInto(std::vector<float>& out, const std::vector<int>& tokens) const
{
    out.assign(static_cast<size_t>(dim), 0.0f);

    const size_t dim_t = static_cast<size_t>(dim);

    // Pré-calcul une seule fois hors boucle.
    const bool has_seq = (seq_embedding.size() == dim_t);
    const bool has_mod = (mod_embedding.size() == dim_t);
    const bool has_mag = (mag_embedding.size() == dim_t);

    #if defined(_MSC_VER)
    #define MIMIR_RESTRICT __restrict
    #else
    #define MIMIR_RESTRICT __restrict__
    #endif

    float* MIMIR_RESTRICT dst = out.data();

    if (tokens.empty()) {
        if (has_seq) {
            const float* MIMIR_RESTRICT s = seq_embedding.data();
            #pragma omp simd
            for (int d = 0; d < dim; ++d) dst[d] += s[d];
        }
        if (has_mod) {
            const float* MIMIR_RESTRICT s = mod_embedding.data();
            #pragma omp simd
            for (int d = 0; d < dim; ++d) dst[d] += s[d];
        }
        if (has_mag) {
            const float* MIMIR_RESTRICT s = mag_embedding.data();
            #pragma omp simd
            for (int d = 0; d < dim; ++d) dst[d] += s[d];
        }
        return;
    }

    float weight_sum = 0.0f;
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
        const int id = tokens[pos];
        if (id <= 0) continue;
        if (id >= vocab_size) continue;

        const float w = (static_cast<int>(pos) < magik_prefix_count && magik_prefix_weight > 0.0f)
                        ? magik_prefix_weight : 1.0f;
        const float* MIMIR_RESTRICT row = token_embeddings.data() + static_cast<size_t>(id) * dim_t;
        #pragma omp simd
        for (int d = 0; d < dim; ++d) dst[d] += row[d] * w;
        weight_sum += w;
    }

    if (weight_sum > 0.0f) {
        const float inv = 1.0f / weight_sum;
        #pragma omp simd
        for (int i = 0; i < dim; ++i) dst[i] *= inv;
    }

    if (has_seq) {
        const float* MIMIR_RESTRICT s = seq_embedding.data();
        #pragma omp simd
        for (int d = 0; d < dim; ++d) dst[d] += s[d];
    }
    if (has_mod) {
        const float* MIMIR_RESTRICT s = mod_embedding.data();
        #pragma omp simd
        for (int d = 0; d < dim; ++d) dst[d] += s[d];
    }
    if (has_mag) {
        const float* MIMIR_RESTRICT s = mag_embedding.data();
        #pragma omp simd
        for (int d = 0; d < dim; ++d) dst[d] += s[d];
    }
}

void ConditioningEncoder::trainOnTextTokens(const std::vector<int> &token_ids, const std::vector<float> &target, float lr)
{
    if (token_ids.empty() || static_cast<int>(target.size()) != dim || lr == 0.0f) return;
    std::cerr << "[encoder] train_text tokens=" << token_ids.size() << " dim=" << dim
              << " lr=" << lr << std::endl;
    const size_t dim_t = static_cast<size_t>(dim);
    const float* MIMIR_RESTRICT tgt = target.data();
    for (int id : token_ids) {
        if (id <= 4) continue; // PAD(0), négatifs, tokens spéciaux (0-4)
        if (id >= vocab_size) continue;
        float* MIMIR_RESTRICT row = token_embeddings.data() + static_cast<size_t>(id) * dim_t;
        // SGD fusé : row += lr * (target - row)  =>  row = (1-lr)*row + lr*target
        #pragma omp simd
        for (int d = 0; d < dim; ++d) {
            row[d] += lr * (tgt[d] - row[d]);
        }
    }
}

#undef MIMIR_RESTRICT

void ConditioningEncoder::fillImageVectorSingleModality(const std::vector<float>& image_feature,
                                                        float lr,
                                                        bool single_modality_training)
{
    if (!single_modality_training) return;
    if (lr == 0.0f || static_cast<int>(image_feature.size()) != dim) return;

    ensureSpecialEmbeddings();
    if (mag_embedding.size() != static_cast<size_t>(dim)) return;

    // Update ONLY mag_embedding in single-modality image training.
    #pragma omp simd
    for (int d = 0; d < dim; ++d) {
        const size_t i = static_cast<size_t>(d);
        mag_embedding[i] += lr * (image_feature[i] - mag_embedding[i]);
    }
}

void ConditioningEncoder::fillTextVectorSingleModality(const std::vector<int>& token_ids,
                                                       int pad_id,
                                                       float lr,
                                                       bool single_modality_training)
{
    if (!single_modality_training) return;
    if (lr == 0.0f || token_ids.empty() || dim <= 0) return;

    ensureSpecialEmbeddings();
    if (seq_embedding.size() != static_cast<size_t>(dim)) return;

    // Build a deterministic text signature from token ids (excluding PAD/specials),
    // then update ONLY seq_embedding toward this signature.
    std::vector<float> target(static_cast<size_t>(dim), 0.0f);
    int valid = 0;
    for (int id : token_ids) {
        if (id <= 4) continue;
        if (pad_id >= 0 && id == pad_id) continue;

        const uint32_t h0 = static_cast<uint32_t>(id) * 0x9e3779b1u;
        const int idx = static_cast<int>((h0 ^ (h0 >> 16)) % static_cast<uint32_t>(dim));
        const float sgn = ((h0 >> 31) & 1u) ? 1.0f : -1.0f;
        target[static_cast<size_t>(idx)] += sgn;
        valid += 1;
    }
    if (valid <= 0) return;

    const float inv = 1.0f / static_cast<float>(valid);
    double ss = 0.0;
    for (int d = 0; d < dim; ++d) {
        target[static_cast<size_t>(d)] *= inv;
        const double v = static_cast<double>(target[static_cast<size_t>(d)]);
        ss += v * v;
    }
    const float norm = static_cast<float>(std::sqrt(std::max(1e-12, ss)));
    for (int d = 0; d < dim; ++d) {
        target[static_cast<size_t>(d)] /= norm;
    }

    #pragma omp simd
    for (int d = 0; d < dim; ++d) {
        const size_t i = static_cast<size_t>(d);
        seq_embedding[i] += lr * (target[i] - seq_embedding[i]);
    }
}