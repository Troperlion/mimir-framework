#pragma once
#include "CudaNativeOps.cuh"
namespace CudaNativeOps {
// Compatibility reference kernels: recurrent time/order dependencies execute
// serially on device. This is Native coverage, not an optimized recurrent engine.
// Separate forward/backward formulas intentionally follow the existing CPU
// contracts, including their legacy gate/state update ordering.
struct SequenceArena {
    float* data;size_t used=0,capacity;
    __device__ float* allocate(size_t n){if(n>capacity-used)return nullptr;float* p=data+used;used+=n;return p;}
};
struct SequenceVector {
    float* values=nullptr;size_t count=0;
    __device__ SequenceVector(float* p,size_t n):values(p),count(n){}
    __device__ SequenceVector(SequenceArena& arena,size_t n,float value=0.f):values(arena.allocate(n)),count(n){for(size_t i=0;i<n;++i)values[i]=value;}
    __device__ size_t size()const{return count;}
    __device__ float* data(){return values;}
    __device__ const float* data()const{return values;}
    __device__ float* begin(){return values;}
    __device__ float& operator[](size_t i){return values[i];}
    __device__ const float& operator[](size_t i)const{return values[i];}
    __device__ void assign(size_t n,float value){for(size_t i=0;i<n;++i)values[i]=value;count=n;}
    __device__ void swap(SequenceVector& other){float* p=values;size_t n=count;values=other.values;count=other.count;other.values=p;other.count=n;}
};
__device__ inline void sequence_copy(const float* first,const float* last,float* output){while(first!=last)*output++=*first++;}
struct SequenceLayer {
    LayerType type_enum;int seq_len,in_features,out_features;bool use_bias;
    float* weights;size_t weights_size;SequenceVector grad_weights;
    __device__ const float* getWeights()const{return weights;}
    __device__ size_t getWeightsSize()const{return weights_size;}
};
__device__ inline float sigmoid_scalar(float v){return 1.f/(1.f+expf(-v));}
__device__ bool recurrent_forward_reference(const SequenceVector& x,SequenceVector* outputs,SequenceLayer& layer,SequenceArena& arena){

                // Pour rester fonctionnel et déterministe, on réutilise l'implémentation CPU de Model (naïve).
                const int T = layer.seq_len;
                const int I = layer.in_features;
                const int H = layer.out_features;
                if (T <= 0 || I <= 0 || H <= 0) return false;
                if (static_cast<int>(x.size()) != T * I) return false;
                const float* W = layer.getWeights();
                if (!W) return false;

                auto sigmoid_scalar = [](float v) -> float { return 1.0f / (1.0f + expf(-v)); };

                if (layer.type_enum == LayerType::RNN) {
                    const bool use_bias = layer.use_bias;
                    const size_t Wih_sz = static_cast<size_t>(H) * static_cast<size_t>(I);
                    const size_t Whh_sz = static_cast<size_t>(H) * static_cast<size_t>(H);
                    const size_t bih_sz = use_bias ? static_cast<size_t>(H) : 0ULL;
                    const size_t bhh_sz = use_bias ? static_cast<size_t>(H) : 0ULL;
                    const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                    if (layer.getWeightsSize() < need) return false;

                    const float* Wih = W;
                    const float* Whh = Wih + Wih_sz;
                    const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
                    const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

                    SequenceVector h_prev(arena, static_cast<size_t>(H), 0.0f);
                    
                    outputs[0].assign(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                    for (int t = 0; t < T; ++t) {
                        const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                        float* ht = &outputs[0][static_cast<size_t>(t) * static_cast<size_t>(H)];
                        for (int h = 0; h < H; ++h) {
                            const float* wih = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                            const float* whh = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                            float sum = 0.0f;
                            for (int i = 0; i < I; ++i) sum += wih[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                            for (int k = 0; k < H; ++k) sum += whh[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                            if (bih) sum += bih[static_cast<size_t>(h)];
                            if (bhh) sum += bhh[static_cast<size_t>(h)];
                            ht[static_cast<size_t>(h)] = tanhf(sum);
                        }
                        sequence_copy(ht, ht + H, h_prev.begin());
                    }
                    return true;
                }

                if (layer.type_enum == LayerType::GRU) {
                    const bool use_bias = layer.use_bias;
                    const size_t Wih_sz = static_cast<size_t>(3 * H) * static_cast<size_t>(I);
                    const size_t Whh_sz = static_cast<size_t>(3 * H) * static_cast<size_t>(H);
                    const size_t bih_sz = use_bias ? static_cast<size_t>(3 * H) : 0ULL;
                    const size_t bhh_sz = use_bias ? static_cast<size_t>(3 * H) : 0ULL;
                    const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                    if (layer.getWeightsSize() < need) return false;

                    const float* Wih = W;
                    const float* Whh = Wih + Wih_sz;
                    const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
                    const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

                    SequenceVector h_prev(arena, static_cast<size_t>(H), 0.0f);
                    SequenceVector r(arena, static_cast<size_t>(H), 0.0f);
                    SequenceVector z(arena, static_cast<size_t>(H), 0.0f);
                    SequenceVector n(arena, static_cast<size_t>(H), 0.0f);

                    
                    outputs[0].assign(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                    for (int t = 0; t < T; ++t) {
                        const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];

                        for (int h = 0; h < H; ++h) {
                            const float* w_ir = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                            const float* w_hr = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                            float sr = 0.0f;
                            for (int i = 0; i < I; ++i) sr += w_ir[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                            for (int k = 0; k < H; ++k) sr += w_hr[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                            if (bih) sr += bih[static_cast<size_t>(h)];
                            if (bhh) sr += bhh[static_cast<size_t>(h)];
                            r[static_cast<size_t>(h)] = sigmoid_scalar(sr);

                            const float* w_iz = Wih + static_cast<size_t>(H + h) * static_cast<size_t>(I);
                            const float* w_hz = Whh + static_cast<size_t>(H + h) * static_cast<size_t>(H);
                            float sz = 0.0f;
                            for (int i = 0; i < I; ++i) sz += w_iz[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                            for (int k = 0; k < H; ++k) sz += w_hz[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                            if (bih) sz += bih[static_cast<size_t>(H + h)];
                            if (bhh) sz += bhh[static_cast<size_t>(H + h)];
                            z[static_cast<size_t>(h)] = sigmoid_scalar(sz);

                            const float* w_in = Wih + static_cast<size_t>(2 * H + h) * static_cast<size_t>(I);
                            const float* w_hn = Whh + static_cast<size_t>(2 * H + h) * static_cast<size_t>(H);
                            float sn = 0.0f;
                            for (int i = 0; i < I; ++i) sn += w_in[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                            for (int k = 0; k < H; ++k) sn += w_hn[static_cast<size_t>(k)] * (r[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)]);
                            if (bih) sn += bih[static_cast<size_t>(2 * H + h)];
                            if (bhh) sn += bhh[static_cast<size_t>(2 * H + h)];
                            n[static_cast<size_t>(h)] = tanhf(sn);
                        }

                        float* ht = &outputs[0][static_cast<size_t>(t) * static_cast<size_t>(H)];
                        for (int h = 0; h < H; ++h) {
                            const float zh = z[static_cast<size_t>(h)];
                            const float nh = n[static_cast<size_t>(h)];
                            const float hp = h_prev[static_cast<size_t>(h)];
                            ht[static_cast<size_t>(h)] = (1.0f - zh) * nh + zh * hp;
                        }
                        sequence_copy(ht, ht + H, h_prev.begin());
                    }
                    return true;
                }

                // LSTM
                {
                    const bool use_bias = layer.use_bias;
                    const size_t Wih_sz = static_cast<size_t>(4 * H) * static_cast<size_t>(I);
                    const size_t Whh_sz = static_cast<size_t>(4 * H) * static_cast<size_t>(H);
                    const size_t bih_sz = use_bias ? static_cast<size_t>(4 * H) : 0ULL;
                    const size_t bhh_sz = use_bias ? static_cast<size_t>(4 * H) : 0ULL;
                    const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                    if (layer.getWeightsSize() < need) return false;

                    const float* Wih = W;
                    const float* Whh = Wih + Wih_sz;
                    const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
                    const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

                    SequenceVector h_prev(arena, static_cast<size_t>(H), 0.0f);
                    SequenceVector c_prev(arena, static_cast<size_t>(H), 0.0f);
                    
                    outputs[0].assign(static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);

                    for (int t = 0; t < T; ++t) {
                        const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                        float* ht = &outputs[0][static_cast<size_t>(t) * static_cast<size_t>(H)];

                        for (int h = 0; h < H; ++h) {
                            auto dot_in = [&](const float* wrow) -> float {
                                float s = 0.0f;
                                for (int i = 0; i < I; ++i) s += wrow[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                                return s;
                            };
                            auto dot_h = [&](const float* wrow) -> float {
                                float s = 0.0f;
                                for (int k = 0; k < H; ++k) s += wrow[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                                return s;
                            };

                            const float* wii = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                            const float* wif = Wih + static_cast<size_t>(H + h) * static_cast<size_t>(I);
                            const float* wig = Wih + static_cast<size_t>(2 * H + h) * static_cast<size_t>(I);
                            const float* wio = Wih + static_cast<size_t>(3 * H + h) * static_cast<size_t>(I);

                            const float* whi = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                            const float* whf = Whh + static_cast<size_t>(H + h) * static_cast<size_t>(H);
                            const float* whg = Whh + static_cast<size_t>(2 * H + h) * static_cast<size_t>(H);
                            const float* who = Whh + static_cast<size_t>(3 * H + h) * static_cast<size_t>(H);

                            float si = dot_in(wii) + dot_h(whi);
                            float sf = dot_in(wif) + dot_h(whf);
                            float sg = dot_in(wig) + dot_h(whg);
                            float so = dot_in(wio) + dot_h(who);
                            if (bih) {
                                si += bih[static_cast<size_t>(h)];
                                sf += bih[static_cast<size_t>(H + h)];
                                sg += bih[static_cast<size_t>(2 * H + h)];
                                so += bih[static_cast<size_t>(3 * H + h)];
                            }
                            if (bhh) {
                                si += bhh[static_cast<size_t>(h)];
                                sf += bhh[static_cast<size_t>(H + h)];
                                sg += bhh[static_cast<size_t>(2 * H + h)];
                                so += bhh[static_cast<size_t>(3 * H + h)];
                            }

                            const float i_gate = sigmoid_scalar(si);
                            const float f_gate = sigmoid_scalar(sf);
                            const float g_gate = tanhf(sg);
                            const float o_gate = sigmoid_scalar(so);

                            const float c = f_gate * c_prev[static_cast<size_t>(h)] + i_gate * g_gate;
                            c_prev[static_cast<size_t>(h)] = c;
                            ht[static_cast<size_t>(h)] = o_gate * tanhf(c);
                        }

                        sequence_copy(ht, ht + H, h_prev.begin());
                    }
                }
                return true;
            
}
__device__ bool recurrent_backward_reference(const SequenceVector* const* inputs,const SequenceVector* const* grad_outputs,
    SequenceVector* grad_inputs,SequenceLayer& layer,SequenceArena& arena){
    switch(layer.type_enum){
case LayerType::LSTM: {

                const SequenceVector& x = *inputs[0];
                const SequenceVector& go = *grad_outputs[0];
                const int T = layer.seq_len;
                const int I = layer.in_features;
                const int H = layer.out_features;
                if (T <= 0 || I <= 0 || H <= 0) return false;
                if (static_cast<int>(x.size()) != T * I) return false;
                if (static_cast<int>(go.size()) != T * H) return false;
                const bool use_bias = layer.use_bias;
                const float* W = layer.getWeights();
                if (!W) return false;

                const size_t Wih_sz = static_cast<size_t>(4 * H) * static_cast<size_t>(I);
                const size_t Whh_sz = static_cast<size_t>(4 * H) * static_cast<size_t>(H);
                const size_t bih_sz = use_bias ? static_cast<size_t>(4 * H) : 0ULL;
                const size_t bhh_sz = use_bias ? static_cast<size_t>(4 * H) : 0ULL;
                const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                if (layer.getWeightsSize() < need) return false;

                const float* Wih = W;
                const float* Whh = Wih + Wih_sz;
                const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
                const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

                SequenceVector i_gate(arena, static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                SequenceVector f_gate(arena, static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                SequenceVector g_gate(arena, static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                SequenceVector o_gate(arena, static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                SequenceVector c_state(arena, static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                SequenceVector h_state(arena, static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);

                SequenceVector h_prev(arena, static_cast<size_t>(H), 0.0f);
                SequenceVector c_prev(arena, static_cast<size_t>(H), 0.0f);
                for (int t = 0; t < T; ++t) {
                    const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                    for (int h = 0; h < H; ++h) {
                        auto dot_in = [&](const float* wrow) -> float {
                            float s = 0.0f;
                            for (int ii = 0; ii < I; ++ii) s += wrow[static_cast<size_t>(ii)] * xt[static_cast<size_t>(ii)];
                            return s;
                        };
                        auto dot_h = [&](const float* wrow) -> float {
                            float s = 0.0f;
                            for (int k = 0; k < H; ++k) s += wrow[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                            return s;
                        };
                        const float* wii = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                        const float* wif = Wih + static_cast<size_t>(H + h) * static_cast<size_t>(I);
                        const float* wig = Wih + static_cast<size_t>(2 * H + h) * static_cast<size_t>(I);
                        const float* wio = Wih + static_cast<size_t>(3 * H + h) * static_cast<size_t>(I);
                        const float* whi = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                        const float* whf = Whh + static_cast<size_t>(H + h) * static_cast<size_t>(H);
                        const float* whg = Whh + static_cast<size_t>(2 * H + h) * static_cast<size_t>(H);
                        const float* who = Whh + static_cast<size_t>(3 * H + h) * static_cast<size_t>(H);

                        float si = dot_in(wii) + dot_h(whi);
                        float sf = dot_in(wif) + dot_h(whf);
                        float sg = dot_in(wig) + dot_h(whg);
                        float so = dot_in(wio) + dot_h(who);
                        if (bih) {
                            si += bih[static_cast<size_t>(h)];
                            sf += bih[static_cast<size_t>(H + h)];
                            sg += bih[static_cast<size_t>(2 * H + h)];
                            so += bih[static_cast<size_t>(3 * H + h)];
                        }
                        if (bhh) {
                            si += bhh[static_cast<size_t>(h)];
                            sf += bhh[static_cast<size_t>(H + h)];
                            sg += bhh[static_cast<size_t>(2 * H + h)];
                            so += bhh[static_cast<size_t>(3 * H + h)];
                        }

                        const float ig = sigmoid_scalar(si);
                        const float fg = sigmoid_scalar(sf);
                        const float gg = tanhf(sg);
                        const float og = sigmoid_scalar(so);
                        const float c = fg * c_prev[static_cast<size_t>(h)] + ig * gg;
                        const float hv = og * tanhf(c);

                        i_gate[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)] = ig;
                        f_gate[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)] = fg;
                        g_gate[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)] = gg;
                        o_gate[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)] = og;
                        c_state[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)] = c;
                        h_state[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)] = hv;
                        c_prev[static_cast<size_t>(h)] = c;
                        h_prev[static_cast<size_t>(h)] = hv;
                    }
                }

                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }
                
                grad_inputs[0].assign(x.size(), 0.0f);

                SequenceVector dh_next(arena, static_cast<size_t>(H), 0.0f);
                SequenceVector dc_next(arena, static_cast<size_t>(H), 0.0f);

                for (int t = T - 1; t >= 0; --t) {
                    const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                    const float* hprev = (t > 0) ? &h_state[static_cast<size_t>(t - 1) * static_cast<size_t>(H)] : nullptr;
                    const float* cprev = (t > 0) ? &c_state[static_cast<size_t>(t - 1) * static_cast<size_t>(H)] : nullptr;

                    SequenceVector dh_prev(arena, static_cast<size_t>(H), 0.0f);
                    SequenceVector dc_prev(arena, static_cast<size_t>(H), 0.0f);

                    for (int h = 0; h < H; ++h) {
                        const size_t th = static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h);
                        const float ig = i_gate[th];
                        const float fg = f_gate[th];
                        const float gg = g_gate[th];
                        const float og = o_gate[th];
                        const float c = c_state[th];
                        const float tanh_c = tanhf(c);
                        const float h_prev_v = hprev ? hprev[static_cast<size_t>(h)] : 0.0f;
                        const float c_prev_v = cprev ? cprev[static_cast<size_t>(h)] : 0.0f;

                        const float dht = go[th] + dh_next[static_cast<size_t>(h)];
                        const float do_gate = dht * tanh_c;
                        const float dc = dc_next[static_cast<size_t>(h)] + dht * og * (1.0f - tanh_c * tanh_c);
                        const float di_gate = dc * gg;
                        const float dg_gate = dc * ig;
                        const float df_gate = dc * c_prev_v;
                        dc_prev[static_cast<size_t>(h)] = dc * fg;

                        const float dai = di_gate * ig * (1.0f - ig);
                        const float daf = df_gate * fg * (1.0f - fg);
                        const float dag = dg_gate * (1.0f - gg * gg);
                        const float dao = do_gate * og * (1.0f - og);

                        const size_t row_i = static_cast<size_t>(h);
                        const size_t row_f = static_cast<size_t>(H + h);
                        const size_t row_g = static_cast<size_t>(2 * H + h);
                        const size_t row_o = static_cast<size_t>(3 * H + h);

                        for (int ii = 0; ii < I; ++ii) {
                            const float xv = xt[static_cast<size_t>(ii)];
                            layer.grad_weights[row_i * static_cast<size_t>(I) + static_cast<size_t>(ii)] += dai * xv;
                            layer.grad_weights[row_f * static_cast<size_t>(I) + static_cast<size_t>(ii)] += daf * xv;
                            layer.grad_weights[row_g * static_cast<size_t>(I) + static_cast<size_t>(ii)] += dag * xv;
                            layer.grad_weights[row_o * static_cast<size_t>(I) + static_cast<size_t>(ii)] += dao * xv;

                            grad_inputs[0][static_cast<size_t>(t) * static_cast<size_t>(I) + static_cast<size_t>(ii)] +=
                                dai * Wih[row_i * static_cast<size_t>(I) + static_cast<size_t>(ii)] +
                                daf * Wih[row_f * static_cast<size_t>(I) + static_cast<size_t>(ii)] +
                                dag * Wih[row_g * static_cast<size_t>(I) + static_cast<size_t>(ii)] +
                                dao * Wih[row_o * static_cast<size_t>(I) + static_cast<size_t>(ii)];
                        }

                        for (int k = 0; k < H; ++k) {
                            const float hp = hprev ? hprev[static_cast<size_t>(k)] : 0.0f;
                            layer.grad_weights[Wih_sz + row_i * static_cast<size_t>(H) + static_cast<size_t>(k)] += dai * hp;
                            layer.grad_weights[Wih_sz + row_f * static_cast<size_t>(H) + static_cast<size_t>(k)] += daf * hp;
                            layer.grad_weights[Wih_sz + row_g * static_cast<size_t>(H) + static_cast<size_t>(k)] += dag * hp;
                            layer.grad_weights[Wih_sz + row_o * static_cast<size_t>(H) + static_cast<size_t>(k)] += dao * hp;

                            dh_prev[static_cast<size_t>(k)] +=
                                dai * Whh[row_i * static_cast<size_t>(H) + static_cast<size_t>(k)] +
                                daf * Whh[row_f * static_cast<size_t>(H) + static_cast<size_t>(k)] +
                                dag * Whh[row_g * static_cast<size_t>(H) + static_cast<size_t>(k)] +
                                dao * Whh[row_o * static_cast<size_t>(H) + static_cast<size_t>(k)];
                        }

                        if (use_bias) {
                            const size_t b0 = Wih_sz + Whh_sz;
                            layer.grad_weights[b0 + row_i] += dai;
                            layer.grad_weights[b0 + row_f] += daf;
                            layer.grad_weights[b0 + row_g] += dag;
                            layer.grad_weights[b0 + row_o] += dao;
                            layer.grad_weights[b0 + bih_sz + row_i] += dai;
                            layer.grad_weights[b0 + bih_sz + row_f] += daf;
                            layer.grad_weights[b0 + bih_sz + row_g] += dag;
                            layer.grad_weights[b0 + bih_sz + row_o] += dao;
                        }
                    }

                    dh_next.swap(dh_prev);
                    dc_next.swap(dc_prev);
                }
                return true;
            
}
case LayerType::GRU: {

                const SequenceVector& x = *inputs[0];
                const SequenceVector& go = *grad_outputs[0];
                const int T = layer.seq_len;
                const int I = layer.in_features;
                const int H = layer.out_features;
                if (T <= 0 || I <= 0 || H <= 0) return false;
                if (static_cast<int>(x.size()) != T * I) return false;
                if (static_cast<int>(go.size()) != T * H) return false;

                const bool use_bias = layer.use_bias;
                const float* W = layer.getWeights();
                if (!W) return false;
                const size_t Wih_sz = static_cast<size_t>(3 * H) * static_cast<size_t>(I);
                const size_t Whh_sz = static_cast<size_t>(3 * H) * static_cast<size_t>(H);
                const size_t bih_sz = use_bias ? static_cast<size_t>(3 * H) : 0ULL;
                const size_t bhh_sz = use_bias ? static_cast<size_t>(3 * H) : 0ULL;
                const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                if (layer.getWeightsSize() < need) return false;
                const float* Wih = W;
                const float* Whh = Wih + Wih_sz;
                const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
                const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

                SequenceVector h_state(arena, static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                SequenceVector r_state(arena, static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                SequenceVector z_state(arena, static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                SequenceVector n_state(arena, static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);

                SequenceVector h_prev(arena, static_cast<size_t>(H), 0.0f);
                for (int t = 0; t < T; ++t) {
                    const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                    for (int h = 0; h < H; ++h) {
                        const float* w_ir = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                        const float* w_hr = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                        float sr = 0.0f;
                        for (int i = 0; i < I; ++i) sr += w_ir[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                        for (int k = 0; k < H; ++k) sr += w_hr[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                        if (bih) sr += bih[static_cast<size_t>(h)];
                        if (bhh) sr += bhh[static_cast<size_t>(h)];
                        const float r = sigmoid_scalar(sr);

                        const float* w_iz = Wih + static_cast<size_t>(H + h) * static_cast<size_t>(I);
                        const float* w_hz = Whh + static_cast<size_t>(H + h) * static_cast<size_t>(H);
                        float sz = 0.0f;
                        for (int i = 0; i < I; ++i) sz += w_iz[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                        for (int k = 0; k < H; ++k) sz += w_hz[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                        if (bih) sz += bih[static_cast<size_t>(H + h)];
                        if (bhh) sz += bhh[static_cast<size_t>(H + h)];
                        const float z = sigmoid_scalar(sz);

                        const float* w_in = Wih + static_cast<size_t>(2 * H + h) * static_cast<size_t>(I);
                        const float* w_hn = Whh + static_cast<size_t>(2 * H + h) * static_cast<size_t>(H);
                        float sn = 0.0f;
                        for (int i = 0; i < I; ++i) sn += w_in[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                        for (int k = 0; k < H; ++k) sn += w_hn[static_cast<size_t>(k)] * (r * h_prev[static_cast<size_t>(k)]);
                        if (bih) sn += bih[static_cast<size_t>(2 * H + h)];
                        if (bhh) sn += bhh[static_cast<size_t>(2 * H + h)];
                        const float n = tanhf(sn);

                        const float hv = (1.0f - z) * n + z * h_prev[static_cast<size_t>(h)];

                        const size_t th = static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h);
                        r_state[th] = r;
                        z_state[th] = z;
                        n_state[th] = n;
                        h_state[th] = hv;
                    }
                    for (int h = 0; h < H; ++h) {
                        h_prev[static_cast<size_t>(h)] = h_state[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)];
                    }
                }

                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }
                
                grad_inputs[0].assign(x.size(), 0.0f);

                SequenceVector dh_next(arena, static_cast<size_t>(H), 0.0f);
                for (int t = T - 1; t >= 0; --t) {
                    const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                    const float* hprev = (t > 0) ? &h_state[static_cast<size_t>(t - 1) * static_cast<size_t>(H)] : nullptr;
                    SequenceVector dh_prev(arena, static_cast<size_t>(H), 0.0f);

                    SequenceVector da_r(arena, static_cast<size_t>(H), 0.0f);
                    SequenceVector da_z(arena, static_cast<size_t>(H), 0.0f);
                    SequenceVector da_n(arena, static_cast<size_t>(H), 0.0f);

                    for (int h = 0; h < H; ++h) {
                        const size_t th = static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h);
                        const float r = r_state[th];
                        const float z = z_state[th];
                        const float n = n_state[th];
                        const float hp = hprev ? hprev[static_cast<size_t>(h)] : 0.0f;
                        const float dht = go[th] + dh_next[static_cast<size_t>(h)];
                        const float dn = dht * (1.0f - z);
                        const float dz = dht * (hp - n);
                        dh_prev[static_cast<size_t>(h)] += dht * z;
                        da_n[static_cast<size_t>(h)] = dn * (1.0f - n * n);
                        da_z[static_cast<size_t>(h)] = dz * z * (1.0f - z);
                        (void)r;
                    }

                    SequenceVector drh(arena, static_cast<size_t>(H), 0.0f);
                    for (int h = 0; h < H; ++h) {
                        const size_t row_n = static_cast<size_t>(2 * H + h);
                        for (int i = 0; i < I; ++i) {
                            const float xv = xt[static_cast<size_t>(i)];
                            layer.grad_weights[row_n * static_cast<size_t>(I) + static_cast<size_t>(i)] += da_n[static_cast<size_t>(h)] * xv;
                            grad_inputs[0][static_cast<size_t>(t) * static_cast<size_t>(I) + static_cast<size_t>(i)] +=
                                da_n[static_cast<size_t>(h)] * Wih[row_n * static_cast<size_t>(I) + static_cast<size_t>(i)];
                        }
                        for (int k = 0; k < H; ++k) {
                            const float hp = hprev ? hprev[static_cast<size_t>(k)] : 0.0f;
                            const float r = r_state[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(k)];
                            layer.grad_weights[Wih_sz + row_n * static_cast<size_t>(H) + static_cast<size_t>(k)] += da_n[static_cast<size_t>(h)] * (r * hp);
                            drh[static_cast<size_t>(k)] += da_n[static_cast<size_t>(h)] * Whh[row_n * static_cast<size_t>(H) + static_cast<size_t>(k)];
                        }
                        if (use_bias) {
                            const size_t b0 = Wih_sz + Whh_sz;
                            layer.grad_weights[b0 + row_n] += da_n[static_cast<size_t>(h)];
                            layer.grad_weights[b0 + bih_sz + row_n] += da_n[static_cast<size_t>(h)];
                        }
                    }

                    for (int k = 0; k < H; ++k) {
                        const float hp = hprev ? hprev[static_cast<size_t>(k)] : 0.0f;
                        const float r = r_state[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(k)];
                        const float dr = drh[static_cast<size_t>(k)] * hp;
                        dh_prev[static_cast<size_t>(k)] += drh[static_cast<size_t>(k)] * r;
                        da_r[static_cast<size_t>(k)] = dr * r * (1.0f - r);
                    }

                    for (int h = 0; h < H; ++h) {
                        const size_t row_r = static_cast<size_t>(h);
                        const size_t row_z = static_cast<size_t>(H + h);

                        for (int i = 0; i < I; ++i) {
                            const float xv = xt[static_cast<size_t>(i)];
                            layer.grad_weights[row_r * static_cast<size_t>(I) + static_cast<size_t>(i)] += da_r[static_cast<size_t>(h)] * xv;
                            layer.grad_weights[row_z * static_cast<size_t>(I) + static_cast<size_t>(i)] += da_z[static_cast<size_t>(h)] * xv;
                            grad_inputs[0][static_cast<size_t>(t) * static_cast<size_t>(I) + static_cast<size_t>(i)] +=
                                da_r[static_cast<size_t>(h)] * Wih[row_r * static_cast<size_t>(I) + static_cast<size_t>(i)] +
                                da_z[static_cast<size_t>(h)] * Wih[row_z * static_cast<size_t>(I) + static_cast<size_t>(i)];
                        }

                        for (int k = 0; k < H; ++k) {
                            const float hp = hprev ? hprev[static_cast<size_t>(k)] : 0.0f;
                            layer.grad_weights[Wih_sz + row_r * static_cast<size_t>(H) + static_cast<size_t>(k)] += da_r[static_cast<size_t>(h)] * hp;
                            layer.grad_weights[Wih_sz + row_z * static_cast<size_t>(H) + static_cast<size_t>(k)] += da_z[static_cast<size_t>(h)] * hp;

                            dh_prev[static_cast<size_t>(k)] +=
                                da_r[static_cast<size_t>(h)] * Whh[row_r * static_cast<size_t>(H) + static_cast<size_t>(k)] +
                                da_z[static_cast<size_t>(h)] * Whh[row_z * static_cast<size_t>(H) + static_cast<size_t>(k)];
                        }

                        if (use_bias) {
                            const size_t b0 = Wih_sz + Whh_sz;
                            layer.grad_weights[b0 + row_r] += da_r[static_cast<size_t>(h)];
                            layer.grad_weights[b0 + row_z] += da_z[static_cast<size_t>(h)];
                            layer.grad_weights[b0 + bih_sz + row_r] += da_r[static_cast<size_t>(h)];
                            layer.grad_weights[b0 + bih_sz + row_z] += da_z[static_cast<size_t>(h)];
                        }
                    }

                    dh_next.swap(dh_prev);
                }
                return true;
            
}
case LayerType::RNN: {

                const SequenceVector& x = *inputs[0];
                const SequenceVector& go = *grad_outputs[0];
                const int T = layer.seq_len;
                const int I = layer.in_features;
                const int H = layer.out_features;
                if (T <= 0 || I <= 0 || H <= 0) return false;
                if (static_cast<int>(x.size()) != T * I) return false;
                if (static_cast<int>(go.size()) != T * H) return false;
                const bool use_bias = layer.use_bias;
                const float* W = layer.getWeights();
                if (!W) return false;
                const size_t Wih_sz = static_cast<size_t>(H) * static_cast<size_t>(I);
                const size_t Whh_sz = static_cast<size_t>(H) * static_cast<size_t>(H);
                const size_t bih_sz = use_bias ? static_cast<size_t>(H) : 0ULL;
                const size_t bhh_sz = use_bias ? static_cast<size_t>(H) : 0ULL;
                const size_t need = Wih_sz + Whh_sz + bih_sz + bhh_sz;
                if (layer.getWeightsSize() < need) return false;

                const float* Wih = W;
                const float* Whh = W + Wih_sz;
                const float* bih = use_bias ? (Whh + Whh_sz) : nullptr;
                const float* bhh = use_bias ? (bih + bih_sz) : nullptr;

                SequenceVector h_state(arena, static_cast<size_t>(T) * static_cast<size_t>(H), 0.0f);
                SequenceVector h_prev(arena, static_cast<size_t>(H), 0.0f);
                for (int t = 0; t < T; ++t) {
                    const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                    float* ht = &h_state[static_cast<size_t>(t) * static_cast<size_t>(H)];
                    for (int h = 0; h < H; ++h) {
                        float sum = 0.0f;
                        const float* wih = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                        const float* whh = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);
                        for (int i = 0; i < I; ++i) sum += wih[static_cast<size_t>(i)] * xt[static_cast<size_t>(i)];
                        for (int k = 0; k < H; ++k) sum += whh[static_cast<size_t>(k)] * h_prev[static_cast<size_t>(k)];
                        if (bih) sum += bih[static_cast<size_t>(h)];
                        if (bhh) sum += bhh[static_cast<size_t>(h)];
                        ht[static_cast<size_t>(h)] = tanhf(sum);
                    }
                    for (int h = 0; h < H; ++h) h_prev[static_cast<size_t>(h)] = ht[static_cast<size_t>(h)];
                }

                if (layer.grad_weights.size() != layer.getWeightsSize()) {
                    layer.grad_weights.assign(layer.getWeightsSize(), 0.0f);
                }

                
                grad_inputs[0].assign(x.size(), 0.0f);
                SequenceVector dh_next(arena, static_cast<size_t>(H), 0.0f);

                for (int t = T - 1; t >= 0; --t) {
                    const float* xt = &x[static_cast<size_t>(t) * static_cast<size_t>(I)];
                    const float* hprev = (t > 0) ? &h_state[static_cast<size_t>(t - 1) * static_cast<size_t>(H)] : nullptr;
                    SequenceVector dh_prev(arena, static_cast<size_t>(H), 0.0f);
                    for (int h = 0; h < H; ++h) {
                        const float h_t = h_state[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)];
                        const float dht = go[static_cast<size_t>(t) * static_cast<size_t>(H) + static_cast<size_t>(h)] + dh_next[static_cast<size_t>(h)];
                        const float dpre = dht * (1.0f - h_t * h_t);

                        const float* wih = Wih + static_cast<size_t>(h) * static_cast<size_t>(I);
                        const float* whh = Whh + static_cast<size_t>(h) * static_cast<size_t>(H);

                        for (int i = 0; i < I; ++i) {
                            layer.grad_weights[static_cast<size_t>(h) * static_cast<size_t>(I) + static_cast<size_t>(i)] += dpre * xt[static_cast<size_t>(i)];
                            grad_inputs[0][static_cast<size_t>(t) * static_cast<size_t>(I) + static_cast<size_t>(i)] += dpre * wih[static_cast<size_t>(i)];
                        }
                        for (int k = 0; k < H; ++k) {
                            const float hp = hprev ? hprev[static_cast<size_t>(k)] : 0.0f;
                            layer.grad_weights[Wih_sz + static_cast<size_t>(h) * static_cast<size_t>(H) + static_cast<size_t>(k)] += dpre * hp;
                            dh_prev[static_cast<size_t>(k)] += dpre * whh[static_cast<size_t>(k)];
                        }
                        if (use_bias) {
                            layer.grad_weights[Wih_sz + Whh_sz + static_cast<size_t>(h)] += dpre;
                            layer.grad_weights[Wih_sz + Whh_sz + bih_sz + static_cast<size_t>(h)] += dpre;
                        }
                    }
                    dh_next.swap(dh_prev);
                }
                return true;
            
}

    default:return false;
    }
}
__global__ void recurrent_kernel(float* x,float* w,float* dy,float* y,float* dw,float* workspace,int* status,
    LayerType type,CudaOperationGeometry::Recurrent g,bool bias,size_t weights_size){
    if(blockIdx.x || threadIdx.x)return;
    SequenceArena arena{workspace,0,g.workspace};SequenceVector xv(x,static_cast<size_t>(g.steps)*g.input),yv(y,dy?static_cast<size_t>(g.steps)*g.input:static_cast<size_t>(g.steps)*g.hidden),gv(dy,static_cast<size_t>(g.steps)*g.hidden);
    SequenceLayer layer{type,g.steps,g.input,g.hidden,bias,w,weights_size,SequenceVector(dw,weights_size)};
    const SequenceVector* in[]={&xv};const SequenceVector* grads[]={&gv};
    *status=dy?recurrent_backward_reference(in,grads,&yv,layer,arena):recurrent_forward_reference(xv,&yv,layer,arena);
}
inline bool recurrent(const std::vector<float>& x,std::vector<float>& y,const Layer& l,const CudaOperationGeometry::Recurrent& g,
    const std::vector<float>* dy=nullptr,std::vector<float>* dw=nullptr){
    Buffer bx,bw,bg,by,bdw,workspace;int* status=nullptr;
    if(!bx.alloc(x.size()) || !bx.upload(x.data(),x.size()) || !bw.alloc(l.getWeightsSize()) || !bw.upload(l.getWeights(),l.getWeightsSize()) || !workspace.alloc(g.workspace))return false;
    y.resize(dy?x.size():static_cast<size_t>(g.steps)*g.hidden);if(!by.alloc(y.size()))return false;
    if(dy && (!bg.alloc(dy->size()) || !bg.upload(dy->data(),dy->size()) || !bdw.alloc(l.getWeightsSize()) || cudaMemset(bdw.ptr,0,l.getWeightsSize()*sizeof(float))!=cudaSuccess))return false;
    if(cudaMalloc(&status,sizeof(int))!=cudaSuccess)return false;
    recurrent_kernel<<<1,1>>>(bx.ptr,bw.ptr,bg.ptr,by.ptr,bdw.ptr,workspace.ptr,status,l.type_enum,g,l.use_bias,l.getWeightsSize());
    int ok=0;const bool done=finished() && cudaMemcpy(&ok,status,sizeof(int),cudaMemcpyDeviceToHost)==cudaSuccess;cudaFree(status);
    if(!done || !ok || !by.download(y.data(),y.size()))return false;
    if(dy){dw->resize(l.getWeightsSize());return bdw.download(dw->data(),dw->size());}return true;
}
} // namespace CudaNativeOps
