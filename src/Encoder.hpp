#ifndef __TENSOR_ENCODER_HPP__
#define __TENSOR_ENCODER_HPP__

#include "Helpers.hpp"
#include "include/json.hpp"
using json = nlohmann::json;


class ConditioningEncoder
{
    public:

        ConditioningEncoder(int d = 64, int Size_Vo = 4096);
        ~ConditioningEncoder();

        void initRandom(uint64_t seed = 0);

        // ensure token_embeddings can cover new_vocab_size; initialize new embeddings randomly
        void ensureVocabSize(size_t new_vocab_size, uint64_t seed = 0xC0FFEEu);

        // set mag embedding from MagicToken (map 8-d embed -> encoder dim)
        void setMagicFromToken(const MagicToken &mt);

        // getters for special embeddings (seq/mod/mag)
        const std::vector<float>& getSeqEmbedding() const { return seq_embedding; }
        const std::vector<float>& getModEmbedding() const { return mod_embedding; }
        const std::vector<float>& getMagEmbedding() const { return mag_embedding; }

        // expose special embeddings (optional)
        void setSeqEmbedding(const std::vector<float> &v);
        void setModEmbedding(const std::vector<float> &v);
        void setMagEmbedding(const std::vector<float> &v);

        // encode: skip PAD(0), add special embeddings for SEQ/MOD/MAG if present
        std::vector<float> encode(const std::vector<int> &tokens, uint32_t /*seed*/ = 0) const;

        // Version sans allocation : écrit dans un buffer pré-alloué (taille >= dim).
        // Préférer cette forme dans les boucles d'entraînement.
        void encodeInto(std::vector<float>& out, const std::vector<int>& tokens) const;

        // train embeddings for token ids toward target embedding
        // special tokens are excluded from direct token_embeddings updates (except normal tokens)
        void trainOnTextTokens(const std::vector<int> &token_ids, const std::vector<float> &target, float lr = 0.01f);

        // Special embeddings: initialization + simple SGD updates
        void ensureSpecialEmbeddings(uint64_t seed = 0x51A5EEDu);

        // Ensure ConditioningEncoder dimension matches a model requirement.
        // Safe for freshly-constructed encoders (no token embeddings allocated).
        // If token embeddings are already present and dim mismatches, throws.
        void ensureDim(int required_dim, uint64_t seed = 0x51A5EEDu);
        void sgdUpdateSpecialEmbeddings(const std::vector<float>& grad_text, float lr,
                           bool update_seq = true,
                           bool update_mod = true,
                           bool update_mag = true);

        // Mono-modality helpers:
        // - image feature updates only MAG vector
        // - text tokens update only SEQ vector
        // They are no-op unless single_modality_training is true.
        void fillImageVectorSingleModality(const std::vector<float>& image_feature,
                           float lr,
                           bool single_modality_training);
        void fillTextVectorSingleModality(const std::vector<int>& token_ids,
                          int pad_id,
                          float lr,
                          bool single_modality_training);

        // expose for checkpointing
        int dim;
        int vocab_size;
        std::vector<float> token_embeddings;

        // MagikTokens: pondération des N premiers tokens de la séquence.
        // Ces tokens (préfixe) représentent typiquement style/thème/artiste/genre/etc.
        int magik_prefix_count = 5;
        float magik_prefix_weight = 2.0f;

        // Serialization (RawFolder + SafeTensors metadata)
        json to_json() const;
        void from_json(const json &j);

    private:
        // special embeddings handled separately
        std::vector<float> seq_embedding;
        std::vector<float> mod_embedding;
        std::vector<float> mag_embedding;
};

#endif //! __TENSOR_ENCODER_HPP__