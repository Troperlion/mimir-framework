#include "Tokenizer.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>
#include <iostream>

Tokenizer::Tokenizer(size_t max_vocab_)
    : maxVocab(max_vocab_)
{
    vocab.reserve(std::min<size_t>(maxVocab, 4096));
    reverse_vocab.reserve(std::min<size_t>(maxVocab, 4096));
    ensureSpecialTokens();
    initializeUnderstandingDictionaries();
}

size_t Tokenizer::getMaxVocab() const {
    return maxVocab;
}

void Tokenizer::setMaxVocab(size_t new_max) {
    const size_t cur = vocab.size();
    maxVocab = std::max(new_max, cur);
    // Best-effort: keep hash tables reasonably sized.
    try {
        vocab.reserve(std::min<size_t>(maxVocab, maxVocab));
        reverse_vocab.reserve(std::min<size_t>(maxVocab, maxVocab));
    } catch (...) {
    }
}

int Tokenizer::addToken(const std::string &tok)
{
    auto it = vocab.find(tok);
    if (it != vocab.end()) return it->second;

    if (vocab.size() >= maxVocab) {
        auto itunk = vocab.find("<UNK>");
        if (itunk != vocab.end()) return itunk->second;
        return 1;  // ID de <UNK>
    }

    int id = static_cast<int>(vocab.size());
    vocab.emplace(tok, id);
    reverse_vocab.emplace(id, tok);
    vocabSize = vocab.size();
    return id;
}

void Tokenizer::ensureVocabFromText(const std::string &text)
{
    // add all words from text to vocab
    auto toks = splitTokens(text);
    for (const auto &w : toks) {
        std::string n = normalizeToken(w);
        if (!n.empty()) addToken(n);
    }
}

int Tokenizer::getPadId() const { auto it = vocab.find("<PAD>"); return it != vocab.end() ? it->second : -1; }
int Tokenizer::getUnkId() const { auto it = vocab.find("<UNK>"); return it != vocab.end() ? it->second : -1; }
int Tokenizer::getSeqId() const { auto it = vocab.find("<SEQ>"); return it != vocab.end() ? it->second : -1; }
int Tokenizer::getModId() const { auto it = vocab.find("<MOD>"); return it != vocab.end() ? it->second : -1; }
int Tokenizer::getMagId() const { auto it = vocab.find("<MAG>"); return it != vocab.end() ? it->second : -1; }
int Tokenizer::getBosId() const { auto it = vocab.find("<BOS>"); return it != vocab.end() ? it->second : -1; }
int Tokenizer::getEosId() const { auto it = vocab.find("<EOS>"); return it != vocab.end() ? it->second : -1; }

size_t Tokenizer::getVocabSize() const { return vocab.size(); }

std::string Tokenizer::getTokenById(int id) const {
    auto it = reverse_vocab.find(id);
    if (it != reverse_vocab.end()) return it->second;
    return "<UNK>";
}

void Tokenizer::ensureSpecialTokens()
{
    // Tokens de contrôle
    const std::vector<std::string> specials = {
        "<PAD>", "<UNK>", "<SEQ>", "<MOD>", "<MAG>", "<BOS>", "<EOS>",
        // Ponctuation courante
        ".", ",", "!", "?", ";", ":", "-", "'", "\"", 
        "(", ")", "[", "]", "{", "}", 
        // Symboles courants
        "+", "-", "*", "/", "=", "<", ">", "&", "|", "~", "@", "#", "$", "%", "^",
        // Espaces spéciaux
        "\n", "\t",
        // Apostrophes françaises courantes
        "l'", "d'", "qu'", "c'", "j'", "m'", "t'", "s'", "n'"
    };
    for (const auto &s : specials) addToken(s);
}

std::vector<std::string> Tokenizer::splitTokens(const std::string &text) const
{
    std::vector<std::string> out;
    std::string cur;
    
    auto isAccentedChar = [](unsigned char c) -> bool {
        return c >= 128;  // UTF-8 multi-byte characters (accents, unicode)
    };
    
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        
        // Gérer les caractères UTF-8 multi-byte (accents)
        if (isAccentedChar(c)) {
            cur.push_back(text[i]);
            // Continuer à lire les bytes UTF-8 suivants
            while (i + 1 < text.size() && (static_cast<unsigned char>(text[i+1]) & 0xC0) == 0x80) {
                cur.push_back(text[++i]);
            }
        }
        // Alphanumeric (inclut '_' pour préserver des tags comme foo_bar)
        else if (std::isalnum(c) || c == '_') {
            cur.push_back(text[i]);
        }
        // Apostrophe contextuelle (l', d', qu', c', j', m', t', s', n')
        else if (c == '\'' && !cur.empty() && i + 1 < text.size() && std::isalpha(text[i+1])) {
            // Vérifier si c'est une contraction française
            std::string lower_cur = cur;
            std::transform(lower_cur.begin(), lower_cur.end(), lower_cur.begin(), ::tolower);
            
            if (lower_cur == "l" || lower_cur == "d" || lower_cur == "qu" || 
                lower_cur == "c" || lower_cur == "j" || lower_cur == "m" || 
                lower_cur == "t" || lower_cur == "s" || lower_cur == "n") {
                // Émettre la contraction comme token séparé
                out.push_back(cur + "'");
                cur.clear();
            } else {
                // Apostrophe interne (ex: "can't", "it's")
                cur.push_back(text[i]);
            }
        }
        // Trait d'union interne (ex: "arc-en-ciel", "peut-être")
        else if (c == '-' && !cur.empty() && i + 1 < text.size() && std::isalpha(text[i+1])) {
            cur.push_back(text[i]);
        }
        // Ponctuation ou whitespace
        else {
            // Émettre le mot courant
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
            
            // Émettre la ponctuation comme token distinct (sauf espaces)
            if (!std::isspace(c)) {
                std::string punct(1, static_cast<char>(c));
                out.push_back(punct);
            }
        }
    }
    
    if (!cur.empty()) out.push_back(cur);
    return out;
}

std::string Tokenizer::normalizeToken(std::string s) const
{
    if (s.empty()) return s;
    
    // Détection de ponctuation pure (garder telle quelle)
    bool isPurePunct = true;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '_' || c >= 128) {  // alnum/'_' ou UTF-8
            isPurePunct = false;
            break;
        }
    }
    if (isPurePunct) return s;  // Ponctuation garde sa casse
    
    // Normalisation pour les mots
    std::string result;
    result.reserve(s.size());
    
    for (size_t i = 0; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        
        // Caractères UTF-8 (accents) : préserver
        if (c >= 128) {
            result.push_back(s[i]);
            // Copier les bytes UTF-8 suivants
            while (i + 1 < s.size() && (static_cast<unsigned char>(s[i+1]) & 0xC0) == 0x80) {
                result.push_back(s[++i]);
            }
        }
        // ASCII lowercase
        else if (std::isupper(c)) {
            result.push_back(std::tolower(c));
        }
        else {
            result.push_back(s[i]);
        }
    }
    
    // Trim ponctuation aux extrémités (mais garder apostrophes/traits d'union internes)
    auto isEdgePunct = [](unsigned char c) -> bool {
        return !(std::isalnum(c) || c == '_') && c < 128 && c != '\'' && c != '-';
    };
    
    while (!result.empty() && isEdgePunct(static_cast<unsigned char>(result.front()))) {
        result.erase(result.begin());
    }
    while (!result.empty() && isEdgePunct(static_cast<unsigned char>(result.back()))) {
        result.pop_back();
    }
    
    return result;
}

std::string Tokenizer::removeAccents(const std::string &text) const
{
    // Table de conversion UTF-8 : accents → ASCII
    static const std::unordered_map<std::string, char> accentMap = {
        // Minuscules
        {"à", 'a'}, {"á", 'a'}, {"â", 'a'}, {"ã", 'a'}, {"ä", 'a'}, {"å", 'a'},
        {"è", 'e'}, {"é", 'e'}, {"ê", 'e'}, {"ë", 'e'},
        {"ì", 'i'}, {"í", 'i'}, {"î", 'i'}, {"ï", 'i'},
        {"ò", 'o'}, {"ó", 'o'}, {"ô", 'o'}, {"õ", 'o'}, {"ö", 'o'},
        {"ù", 'u'}, {"ú", 'u'}, {"û", 'u'}, {"ü", 'u'},
        {"ç", 'c'}, {"ñ", 'n'}, {"ý", 'y'}, {"ÿ", 'y'},
        {"æ", 'a'}, {"œ", 'o'},
        // Majuscules
        {"À", 'A'}, {"Á", 'A'}, {"Â", 'A'}, {"Ã", 'A'}, {"Ä", 'A'}, {"Å", 'A'},
        {"È", 'E'}, {"É", 'E'}, {"Ê", 'E'}, {"Ë", 'E'},
        {"Ì", 'I'}, {"Í", 'I'}, {"Î", 'I'}, {"Ï", 'I'},
        {"Ò", 'O'}, {"Ó", 'O'}, {"Ô", 'O'}, {"Õ", 'O'}, {"Ö", 'O'},
        {"Ù", 'U'}, {"Ú", 'U'}, {"Û", 'U'}, {"Ü", 'U'},
        {"Ç", 'C'}, {"Ñ", 'N'}, {"Ý", 'Y'},
        {"Æ", 'A'}, {"Œ", 'O'}
    };
    
    std::string result;
    result.reserve(text.size());
    
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        
        // Caractère UTF-8 (potentiellement accent)
        if (c >= 128) {
            // Extraire le caractère UTF-8 complet (2-4 bytes)
            std::string utf8_char;
            utf8_char.push_back(text[i]);
            while (i + 1 < text.size() && (static_cast<unsigned char>(text[i+1]) & 0xC0) == 0x80) {
                utf8_char.push_back(text[++i]);
            }
            
            // Chercher conversion
            auto it = accentMap.find(utf8_char);
            if (it != accentMap.end()) {
                result.push_back(it->second);
            } else {
                result += utf8_char;  // Garder si non trouvé
            }
        } else {
            result.push_back(text[i]);
        }
    }
    
    return result;
}

bool Tokenizer::containsAccents(const std::string &text) const
{
    for (unsigned char c : text) {
        if (c >= 128) return true;  // UTF-8 détecté
    }
    return false;
}

std::string Tokenizer::escapePunctuation(const std::string &text) const
{
    // Échapper les caractères spéciaux pour regex
    static const std::string special_chars = ".^$*+?()[{\\|";
    std::string result;
    result.reserve(text.size() * 2);
    
    for (char c : text) {
        if (special_chars.find(c) != std::string::npos) {
            result.push_back('\\');
        }
        result.push_back(c);
    }
    
    return result;
}

std::vector<int> Tokenizer::tokenize(const std::string &text) const
{
    std::vector<int> out;
    auto toks = splitTokens(text);
    int unk = getUnkId();
    if (unk < 0) unk = 0;

    const int bos = getBosId();
    const int eos = getEosId();
    const bool use_bos_eos = (bos >= 0 && eos >= 0);

    out.reserve(toks.size() + (use_bos_eos ? 2 : 0));
    if (use_bos_eos) out.push_back(bos);

    for (const auto &w : toks) {
        std::string n = normalizeToken(w);
        if (n.empty()) continue;
        auto it = vocab.find(n);
        if (it != vocab.end()) out.push_back(it->second);
        else out.push_back(unk);
    }

    if (use_bos_eos) out.push_back(eos);
    return out;
}

std::vector<int> Tokenizer::tokenizeEnsure(const std::string &text)
{
    std::vector<int> out;
    auto toks = splitTokens(text);

    const int bos = getBosId();
    const int eos = getEosId();
    const bool use_bos_eos = (bos >= 0 && eos >= 0);

    out.reserve(toks.size() + (use_bos_eos ? 2 : 0));
    if (use_bos_eos) out.push_back(bos);

    for (const auto &w : toks) {
        std::string n = normalizeToken(w);
        if (n.empty()) continue;
        int id = addToken(n);
        out.push_back(id);
    }

    if (use_bos_eos) out.push_back(eos);
    return out;
}

std::string Tokenizer::decode(const std::vector<int> &ids) const
{
    std::ostringstream ss;
    
    auto isPunctuation = [](const std::string& s) -> bool {
        if (s.empty()) return false;
        if (s.size() > 1) return false;  // Multi-char n'est pas ponctuation simple
        unsigned char c = s[0];
        return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) || 
               (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
    };
    
    auto needsSpaceBefore = [](const std::string& s) -> bool {
        return s != "." && s != "," && s != "!" && s != "?" && 
               s != ";" && s != ":" && s != ")" && s != "]" && 
               s != "}" && s != "'" && s != "\"" && 
               !s.empty() && s[0] != '\'';
    };
    
    auto needsSpaceAfter = [](const std::string& s) -> bool {
        return s != "(" && s != "[" && s != "{" && s != "'" && 
               s != "\"" && !s.empty() && s.back() != '\'';
    };
    
    std::string prev;
    for (int id : ids) {
        std::string token = getTokenById(id);
        
        // Ajouter espace intelligent
        if (!prev.empty()) {
            bool curr_is_punct = isPunctuation(token);
            bool prev_is_punct = isPunctuation(prev);
            
            // Pas d'espace avant ponctuation fermante ou après ouvrante
            if (!(curr_is_punct && !needsSpaceBefore(token)) &&
                !(prev_is_punct && !needsSpaceAfter(prev))) {
                ss << ' ';
            }
        }
        
        ss << token;
        prev = token;
    }
    
    return ss.str();
}

json Tokenizer::to_json() const
{
    json j;
    j["maxVocab"] = maxVocab;
    j["vocabSize"] = vocabSize;
    
    try {
        // Sauvegarder vocab directement (token -> id)
        json vocab_obj = json::object();
        for (const auto& [token, id] : vocab) {
            // IMPORTANT: keep tokens verbatim. JSON dumping will escape control chars
            // (e.g. "\n", "\t") in a reversible way. Replacing them by spaces
            // would introduce key collisions and lose vocabulary entries.
            vocab_obj[token] = id;
        }
        j["vocab"] = std::move(vocab_obj);
    } catch (const std::exception &e) {
        j["vocab"] = json::object();
        j["vocab_error"] = std::string("to_json exception: ") + e.what();
    } catch (...) {
        j["vocab"] = json::object();
        j["vocab_error"] = "to_json unknown exception";
    }
    return j;
}

void Tokenizer::from_json(const json &j)
{
    if (j.contains("maxVocab")) maxVocab = j["maxVocab"].get<size_t>();
    
    // Support ancien format id2token (rétrocompatibilité)
    if (j.contains("id2token")) {
        try {
            auto id2token_arr = j["id2token"].get<std::vector<std::string>>();
            vocab.clear();
            reverse_vocab.clear();
            vocab.reserve(id2token_arr.size());
            reverse_vocab.reserve(id2token_arr.size());
            for (size_t i = 0; i < id2token_arr.size(); ++i) {
                vocab.emplace(id2token_arr[i], static_cast<int>(i));
                reverse_vocab.emplace(static_cast<int>(i), id2token_arr[i]);
            }
            vocabSize = vocab.size();
        } catch (...) {
            // ignore malformed json
        }
    }
    
    // Nouveau format vocab (token -> id)
    if (j.contains("vocab")) {
        try {
            auto vocab_obj = j["vocab"].get<std::unordered_map<std::string, int>>();
            vocab = std::move(vocab_obj);
            reverse_vocab.clear();
            reverse_vocab.reserve(vocab.size());
            for (const auto& [token, id] : vocab) {
                reverse_vocab.emplace(id, token);
            }
            vocabSize = vocab.size();
        } catch (...) {
            // ignore malformed json
        }
    }
}

void Tokenizer::setMaxSequenceLength(int max_len) {
    maxSequenceLength = max_len;
}

std::vector<int> Tokenizer::padSequence(const std::vector<int>& tokens, int target_len) const {
    int len = (target_len < 0) ? maxSequenceLength : target_len;

    if (len <= 0) return {};

    const int pad_id = getPadId();
    const int bos = getBosId();
    const int eos = getEosId();
    const bool use_bos_eos = (len >= 2 && bos >= 0 && eos >= 0);

    // Travail sur un contenu sans BOS/EOS, puis on reconstruit.
    size_t start = 0;
    size_t end = tokens.size();
    if (use_bos_eos) {
        if (start < end && tokens[start] == bos) ++start;
        if (end > start && tokens[end - 1] == eos) --end;
    }

    std::vector<int> content;
    if (end > start) {
        content.assign(tokens.begin() + static_cast<std::ptrdiff_t>(start), tokens.begin() + static_cast<std::ptrdiff_t>(end));
    }

    std::vector<int> padded;
    padded.reserve(static_cast<size_t>(len));

    if (use_bos_eos) {
        padded.push_back(bos);
        const int content_cap = std::max(0, len - 2);
        if (static_cast<int>(content.size()) > content_cap) {
            content.resize(static_cast<size_t>(content_cap));
        }
        padded.insert(padded.end(), content.begin(), content.end());
        while (static_cast<int>(padded.size()) < (len - 1)) {
            padded.push_back(pad_id);
        }
        padded.push_back(eos);
        return padded;
    }

    // Fallback: padding/troncature classique.
    padded = tokens;
    
    // Tronquer si trop long
    if (padded.size() > static_cast<size_t>(len)) {
        padded.resize(len);
    }
    
    // Padding si trop court
    while (padded.size() < static_cast<size_t>(len)) {
        padded.push_back(pad_id);
    }
    
    return padded;
}

std::vector<std::vector<int>> Tokenizer::batchTokenize(
    const std::vector<std::string>& texts, 
    int max_len
) {
    std::vector<std::vector<int>> batch;
    batch.reserve(texts.size());
    
    for (const auto& text : texts) {
        auto tokens = tokenizeEnsure(text);
        auto padded = padSequence(tokens, max_len);
        batch.push_back(padded);
    }
    
    return batch;
}

void Tokenizer::printVocabStats() const {
    std::cout << "\n📊 Statistiques du Vocabulaire" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "  Taille actuelle : " << vocabSize << " / " << maxVocab << std::endl;
    std::cout << "  Tokens spéciaux : 7 (<PAD>, <UNK>, <SEQ>, <MOD>, <MAG>, <BOS>, <EOS>)" << std::endl;
    std::cout << "  Tokens utilisateur : " << (vocabSize > 7 ? (vocabSize - 7) : 0) << std::endl;
    std::cout << "  Max séquence : " << maxSequenceLength << std::endl;
    
    // Afficher les 10 premiers tokens
    std::cout << "\n  Premiers tokens :" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(10, vocab.size()); ++i) {
        std::cout << "    [" << i << "] " << getTokenById(static_cast<int>(i)) << std::endl;
    }
}

std::unordered_map<std::string, int> Tokenizer::getTokenFrequencies(
    const std::string& text
) const {
    std::unordered_map<std::string, int> freqs;
    auto tokens = splitTokens(text);
    
    for (const auto& tok : tokens) {
        std::string normalized = normalizeToken(tok);
        if (!normalized.empty()) {
            freqs[normalized]++;
        }
    }
    
    return freqs;
}

void Tokenizer::learnBPEFromCorpus(
    const std::vector<std::string>& corpus, 
    int num_merges
) {
    std::cout << "\n🔤 Apprentissage BPE..." << std::endl;
    
    // 1. Compter les paires de tokens
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> pair_counts;
    
    for (const auto& text : corpus) {
        auto tokens = splitTokens(text);
        for (size_t i = 0; i + 1 < tokens.size(); ++i) {
            std::string t1 = normalizeToken(tokens[i]);
            std::string t2 = normalizeToken(tokens[i + 1]);
            if (!t1.empty() && !t2.empty()) {
                pair_counts[{t1, t2}]++;
            }
        }
    }
    
    // 2. Effectuer les merges
    for (int merge = 0; merge < num_merges && !pair_counts.empty(); ++merge) {
        // Trouver la paire la plus fréquente
        auto max_pair = std::max_element(
            pair_counts.begin(), 
            pair_counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; }
        );
        
        if (max_pair->second < 2) break;  // Arrêter si fréquence < 2
        
        // Merger la paire
        std::string merged = max_pair->first.first + max_pair->first.second;
        addToken(merged);
        bpeMerges[max_pair->first] = merge;
        
        if ((merge + 1) % 100 == 0) {
            std::cout << "  Merge " << (merge + 1) << "/" << num_merges 
                      << " : " << max_pair->first.first << " + " 
                      << max_pair->first.second << " → " << merged 
                      << " (freq: " << max_pair->second << ")" << std::endl;
        }
        
        pair_counts.erase(max_pair);
    }
    
    std::cout << "  ✓ BPE terminé : " << bpeMerges.size() << " merges appris" << std::endl;
}

std::vector<int> Tokenizer::tokenizeBPE(const std::string& text) const {
    if (bpeMerges.empty()) {
        // Fallback sur tokenization standard
        return tokenize(text);
    }
    
    std::vector<std::string> tokens = splitTokens(text);
    std::vector<int> ids;
    ids.reserve(tokens.size());
    
    for (auto& tok : tokens) {
        std::string normalized = normalizeToken(tok);
        if (normalized.empty()) continue;
        
        // Chercher dans vocab (avec merges BPE appliqués)
        auto it = vocab.find(normalized);
        if (it != vocab.end()) {
            ids.push_back(it->second);
        } else {
            ids.push_back(getUnkId());
        }
    }
    
    return ids;
}

void Tokenizer::initializeUnderstandingDictionaries() {
    // Stop words (mots vides courants)
    stopWords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                 "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
                 "been", "being", "have", "has", "had", "do", "does", "did", "will",
                 "would", "should", "could", "may", "might", "must", "can", "this",
                 "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"};
    
    // Verbes d'action courants
    actionVerbs = {"create", "make", "draw", "paint", "generate", "show", "display",
                   "render", "visualize", "depict", "illustrate", "portray", "design",
                   "compose", "produce", "build", "craft", "form", "shape", "construct"};
    
    // Types d'entités courantes
    entityTypes["red"] = "color"; entityTypes["blue"] = "color"; entityTypes["green"] = "color";
    entityTypes["yellow"] = "color"; entityTypes["orange"] = "color"; entityTypes["purple"] = "color";
    entityTypes["pink"] = "color"; entityTypes["white"] = "color"; entityTypes["black"] = "color";
    entityTypes["brown"] = "color"; entityTypes["gray"] = "color"; entityTypes["grey"] = "color";
    
    entityTypes["flower"] = "object"; entityTypes["rose"] = "object"; entityTypes["tree"] = "object";
    entityTypes["leaf"] = "object"; entityTypes["petal"] = "object"; entityTypes["stem"] = "object";
    entityTypes["garden"] = "place"; entityTypes["forest"] = "place"; entityTypes["mountain"] = "place";
    entityTypes["ocean"] = "place"; entityTypes["sky"] = "place"; entityTypes["sun"] = "object";
    entityTypes["moon"] = "object"; entityTypes["star"] = "object"; entityTypes["cloud"] = "object";
    
    entityTypes["beautiful"] = "modifier"; entityTypes["pretty"] = "modifier";
    entityTypes["stunning"] = "modifier"; entityTypes["gorgeous"] = "modifier";
    entityTypes["bright"] = "modifier"; entityTypes["dark"] = "modifier";
    entityTypes["large"] = "modifier"; entityTypes["small"] = "modifier";
    entityTypes["big"] = "modifier"; entityTypes["tiny"] = "modifier";
    
    // Scores de sentiment (-1.0 = négatif, 0.0 = neutre, 1.0 = positif)
    sentimentScores["beautiful"] = 0.9f; sentimentScores["ugly"] = -0.9f;
    sentimentScores["good"] = 0.7f; sentimentScores["bad"] = -0.7f;
    sentimentScores["great"] = 0.8f; sentimentScores["terrible"] = -0.8f;
    sentimentScores["amazing"] = 0.9f; sentimentScores["awful"] = -0.9f;
    sentimentScores["wonderful"] = 0.9f; sentimentScores["horrible"] = -0.9f;
    sentimentScores["excellent"] = 0.95f; sentimentScores["poor"] = -0.6f;
    sentimentScores["stunning"] = 0.9f; sentimentScores["disgusting"] = -0.9f;
    sentimentScores["bright"] = 0.5f; sentimentScores["dark"] = -0.3f;
    sentimentScores["happy"] = 0.8f; sentimentScores["sad"] = -0.7f;
    sentimentScores["fresh"] = 0.6f; sentimentScores["rotten"] = -0.8f;
}

Tokenizer::TextUnderstanding Tokenizer::analyzeText(const std::string& text) const {
    TextUnderstanding understanding;
    
    auto tokens = splitTokens(text);
    std::vector<std::string> normalizedTokens;
    
    for (const auto& tok : tokens) {
        std::string normalized = normalizeToken(tok);
        if (!normalized.empty() && stopWords.find(normalized) == stopWords.end()) {
            normalizedTokens.push_back(normalized);
        }
    }
    
    // Détection des entités et classificiation
    for (const auto& tok : normalizedTokens) {
        auto it = entityTypes.find(tok);
        if (it != entityTypes.end()) {
            if (it->second == "object" || it->second == "place") {
                understanding.entities.push_back(tok);
            } else if (it->second == "modifier") {
                understanding.modifiers.push_back(tok);
            }
        }
        
        if (actionVerbs.find(tok) != actionVerbs.end()) {
            understanding.actions.push_back(tok);
        }
    }
    
    // Sujet principal (première entité significative ou mot le plus fréquent)
    if (!understanding.entities.empty()) {
        understanding.mainSubject = understanding.entities[0];
    } else if (!normalizedTokens.empty()) {
        // Trouver le mot le plus long (souvent le plus significatif)
        understanding.mainSubject = *std::max_element(
            normalizedTokens.begin(), normalizedTokens.end(),
            [](const std::string& a, const std::string& b) { return a.length() < b.length(); }
        );
    }
    
    // Analyse de sentiment
    understanding.sentiment = analyzeSentiment(text);
    
    // Contexte
    understanding.context = inferContext(text);
    
    // Complexité
    understanding.complexity = static_cast<int>(computeTextComplexity(text));
    
    // Relations (mots adjacents significatifs)
    for (size_t i = 0; i + 1 < normalizedTokens.size(); ++i) {
        const auto& t1 = normalizedTokens[i];
        const auto& t2 = normalizedTokens[i + 1];
        
        // Créer une relation si les deux mots sont des entités ou modificateur + entité
        bool t1_entity = entityTypes.find(t1) != entityTypes.end();
        bool t2_entity = entityTypes.find(t2) != entityTypes.end();
        
        if (t1_entity && t2_entity) {
            understanding.relations.push_back({t1, t2});
        }
    }
    
    return understanding;
}

std::vector<std::string> Tokenizer::extractKeywords(const std::string& text, int topN) const {
    auto freqs = getTokenFrequencies(text);
    
    // Filtrer les stop words et trier par fréquence
    std::vector<std::pair<std::string, int>> filtered;
    for (const auto& [word, freq] : freqs) {
        if (stopWords.find(word) == stopWords.end()) {
            filtered.push_back({word, freq});
        }
    }
    
    std::sort(filtered.begin(), filtered.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::vector<std::string> keywords;
    for (int i = 0; i < topN && i < static_cast<int>(filtered.size()); ++i) {
        keywords.push_back(filtered[i].first);
    }
    
    return keywords;
}

std::vector<std::string> Tokenizer::detectEntities(const std::string& text) const {
    std::vector<std::string> entities;
    auto tokens = splitTokens(text);
    
    for (const auto& tok : tokens) {
        std::string normalized = normalizeToken(tok);
        auto it = entityTypes.find(normalized);
        
        if (it != entityTypes.end() && 
            (it->second == "object" || it->second == "place" || it->second == "color")) {
            entities.push_back(normalized);
        }
    }
    
    return entities;
}

std::string Tokenizer::inferContext(const std::string& text) const {
    auto entities = detectEntities(text);
    
    // Compter les catégories d'entités
    std::unordered_map<std::string, int> categoryCount;
    for (const auto& entity : entities) {
        auto it = entityTypes.find(entity);
        if (it != entityTypes.end()) {
            categoryCount[it->second]++;
        }
    }
    
    // Déterminer le contexte dominant
    if (categoryCount["color"] > 0 && categoryCount["object"] > 0) {
        return "visual_description";
    } else if (categoryCount["place"] > 0) {
        return "location";
    } else if (categoryCount["object"] > 0) {
        return "object_focused";
    } else if (!entities.empty()) {
        return "descriptive";
    }
    
    return "general";
}

float Tokenizer::computeTextComplexity(const std::string& text) const {
    auto tokens = splitTokens(text);
    if (tokens.empty()) return 0.0f;
    
    // Facteurs de complexité:
    // 1. Longueur moyenne des mots
    float avgLength = 0.0f;
    for (const auto& tok : tokens) {
        avgLength += tok.length();
    }
    avgLength /= tokens.size();
    
    // 2. Diversité du vocabulaire (ratio tokens uniques / total)
    std::unordered_set<std::string> unique;
    for (const auto& tok : tokens) {
        unique.insert(normalizeToken(tok));
    }
    float diversity = static_cast<float>(unique.size()) / tokens.size();
    
    // 3. Nombre de tokens
    float lengthFactor = std::min(10.0f, static_cast<float>(tokens.size()) / 5.0f);
    
    // Combiner les facteurs (0-10)
    float complexity = (avgLength * 0.5f) + (diversity * 5.0f) + (lengthFactor * 0.3f);
    
    return std::min(10.0f, std::max(0.0f, complexity));
}

std::unordered_map<std::string, float> Tokenizer::analyzeSentiment(const std::string& text) const {
    std::unordered_map<std::string, float> sentiment;
    auto tokens = splitTokens(text);
    
    for (const auto& tok : tokens) {
        std::string normalized = normalizeToken(tok);
        auto it = sentimentScores.find(normalized);
        
        if (it != sentimentScores.end()) {
            sentiment[normalized] = it->second;
        }
    }
    
    return sentiment;
}

