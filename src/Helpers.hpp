#ifndef __TENSOR_HELPERS_HPP__
#define __TENSOR_HELPERS_HPP__
#include "runtimes/fpga/FpgaIo.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <cstdint>
#include <iostream>
#include <filesystem>
#include <algorithm> // std::clamp
#include <cctype>    // std::tolower
#include <cmath>
#include <iomanip>   // dump formaté
#include <numeric>   // std::accumulate
#include <regex>
#include <unordered_set>
#include <unordered_map>
#include <stdio.h>
#include <cstring>
#include <iterator>
#include <optional>
#include <limits>

namespace fs = std::filesystem;

#include "include/json.hpp"
using json = nlohmann::json;

#include "Sha256.hpp"
#include "DatasetAnnotation.hpp"

#ifdef MIMIR_ENABLE_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/version.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

static inline bool ffmpegDecodeAudioToPcm(const std::string &audio_file, std::vector<uint8_t> &out, std::string *err = nullptr)
{
    out.clear();

    AVFormatContext *format = nullptr;
    AVCodecContext *codec = nullptr;
    SwrContext *swr = nullptr;
    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    if (!packet || !frame) {
        if (err) *err = "ffmpeg allocation failed";
        if (packet) av_packet_free(&packet);
        if (frame) av_frame_free(&frame);
        return false;
    }

    auto cleanup = [&]() {
        if (swr) swr_free(&swr);
        if (codec) avcodec_free_context(&codec);
        if (format) avformat_close_input(&format);
        av_frame_free(&frame);
        av_packet_free(&packet);
    };

    if (avformat_open_input(&format, audio_file.c_str(), nullptr, nullptr) < 0) {
        if (err) *err = "avformat_open_input failed";
        cleanup();
        return false;
    }
    if (avformat_find_stream_info(format, nullptr) < 0) {
        if (err) *err = "avformat_find_stream_info failed";
        cleanup();
        return false;
    }

    const int stream_index = av_find_best_stream(format, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (stream_index < 0) {
        if (err) *err = "no audio stream found";
        cleanup();
        return false;
    }

    AVStream *stream = format->streams[stream_index];
    const AVCodec *decoder = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!decoder) {
        if (err) *err = "audio decoder not found";
        cleanup();
        return false;
    }

    codec = avcodec_alloc_context3(decoder);
    if (!codec) {
        if (err) *err = "avcodec_alloc_context3 failed";
        cleanup();
        return false;
    }
    if (avcodec_parameters_to_context(codec, stream->codecpar) < 0 || avcodec_open2(codec, decoder, nullptr) < 0) {
        if (err) *err = "avcodec open failed";
        cleanup();
        return false;
    }

#if LIBAVUTIL_VERSION_MAJOR >= 57
    AVChannelLayout output_channel_layout = AV_CHANNEL_LAYOUT_STEREO;
    AVChannelLayout input_channel_layout = codec->ch_layout;
    if (input_channel_layout.nb_channels <= 0) {
        av_channel_layout_default(&input_channel_layout, 2);
    }
    const int swr_alloc_result = swr_alloc_set_opts2(
        &swr,
        &output_channel_layout,
        AV_SAMPLE_FMT_S16,
        48000,
        &input_channel_layout,
        codec->sample_fmt,
        codec->sample_rate,
        0,
        nullptr
    );
#else
    int64_t input_channel_layout = codec->channel_layout;
    if (input_channel_layout == 0) {
        const int input_channels = codec->channels > 0 ? codec->channels : 2;
        input_channel_layout = av_get_default_channel_layout(input_channels);
    }
    swr = swr_alloc_set_opts(
        nullptr,
        AV_CH_LAYOUT_STEREO,
        AV_SAMPLE_FMT_S16,
        48000,
        input_channel_layout,
        codec->sample_fmt,
        codec->sample_rate,
        0,
        nullptr
    );
    const int swr_alloc_result = swr ? 0 : AVERROR(ENOMEM);
#endif
    if (swr_alloc_result < 0 || !swr || swr_init(swr) < 0) {
        if (err) *err = "swr_init failed";
        cleanup();
        return false;
    }

    std::vector<uint8_t> pcm;
    while (av_read_frame(format, packet) >= 0) {
        if (packet->stream_index != stream_index) {
            av_packet_unref(packet);
            continue;
        }
        if (avcodec_send_packet(codec, packet) < 0) {
            av_packet_unref(packet);
            continue;
        }
        av_packet_unref(packet);

        while (avcodec_receive_frame(codec, frame) == 0) {
            const int out_samples = av_rescale_rnd(
                swr_get_delay(swr, codec->sample_rate) + frame->nb_samples,
                48000,
                codec->sample_rate,
                AV_ROUND_UP
            );
            std::vector<uint8_t> buffer((size_t)out_samples * 2ULL * sizeof(int16_t));
            uint8_t *out_data[1] = { buffer.data() };
            const int converted = swr_convert(
                swr,
                out_data,
                out_samples,
                (const uint8_t **)frame->extended_data,
                frame->nb_samples
            );
            if (converted > 0) {
                buffer.resize((size_t)converted * 2ULL * sizeof(int16_t));
                pcm.insert(pcm.end(), buffer.begin(), buffer.end());
            }
        }
    }

    if (avcodec_send_packet(codec, nullptr) == 0) {
        while (avcodec_receive_frame(codec, frame) == 0) {
            const int out_samples = av_rescale_rnd(
                swr_get_delay(swr, codec->sample_rate) + frame->nb_samples,
                48000,
                codec->sample_rate,
                AV_ROUND_UP
            );
            std::vector<uint8_t> buffer((size_t)out_samples * 2ULL * sizeof(int16_t));
            uint8_t *out_data[1] = { buffer.data() };
            const int converted = swr_convert(
                swr,
                out_data,
                out_samples,
                (const uint8_t **)frame->extended_data,
                frame->nb_samples
            );
            if (converted > 0) {
                buffer.resize((size_t)converted * 2ULL * sizeof(int16_t));
                pcm.insert(pcm.end(), buffer.begin(), buffer.end());
            }
        }
    }

    if (pcm.empty()) {
        if (err) *err = "audio decode produced no samples";
        cleanup();
        return false;
    }

    out = std::move(pcm);
    cleanup();
    return true;
}

static inline bool ffmpegDecodeVideoToRgbFirstFrame(const std::string &video_file, std::vector<uint8_t> &out, int &out_w, int &out_h, std::string *err = nullptr)
{
    out.clear();
    out_w = 0;
    out_h = 0;

    AVFormatContext *format = nullptr;
    AVCodecContext *codec = nullptr;
    SwsContext *sws = nullptr;
    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    AVFrame *rgb = av_frame_alloc();
    if (!packet || !frame || !rgb) {
        if (err) *err = "ffmpeg allocation failed";
        if (packet) av_packet_free(&packet);
        if (frame) av_frame_free(&frame);
        if (rgb) av_frame_free(&rgb);
        return false;
    }

    auto cleanup = [&]() {
        if (sws) sws_freeContext(sws);
        if (codec) avcodec_free_context(&codec);
        if (format) avformat_close_input(&format);
        av_frame_free(&frame);
        av_frame_free(&rgb);
        av_packet_free(&packet);
    };

    if (avformat_open_input(&format, video_file.c_str(), nullptr, nullptr) < 0) {
        if (err) *err = "avformat_open_input failed";
        cleanup();
        return false;
    }
    if (avformat_find_stream_info(format, nullptr) < 0) {
        if (err) *err = "avformat_find_stream_info failed";
        cleanup();
        return false;
    }

    const int stream_index = av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (stream_index < 0) {
        if (err) *err = "no video stream found";
        cleanup();
        return false;
    }

    AVStream *stream = format->streams[stream_index];
    const AVCodec *decoder = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!decoder) {
        if (err) *err = "video decoder not found";
        cleanup();
        return false;
    }

    codec = avcodec_alloc_context3(decoder);
    if (!codec) {
        if (err) *err = "avcodec_alloc_context3 failed";
        cleanup();
        return false;
    }
    if (avcodec_parameters_to_context(codec, stream->codecpar) < 0 || avcodec_open2(codec, decoder, nullptr) < 0) {
        if (err) *err = "avcodec open failed";
        cleanup();
        return false;
    }

    out_w = codec->width > 0 ? codec->width : stream->codecpar->width;
    out_h = codec->height > 0 ? codec->height : stream->codecpar->height;
    if (out_w <= 0 || out_h <= 0) {
        if (err) *err = "invalid video dimensions";
        cleanup();
        return false;
    }

    rgb->format = AV_PIX_FMT_RGB24;
    rgb->width = out_w;
    rgb->height = out_h;
    if (av_frame_get_buffer(rgb, 32) < 0) {
        if (err) *err = "av_frame_get_buffer failed";
        cleanup();
        return false;
    }

    sws = sws_getContext(
        out_w,
        out_h,
        codec->pix_fmt,
        out_w,
        out_h,
        AV_PIX_FMT_RGB24,
        SWS_BICUBIC,
        nullptr,
        nullptr,
        nullptr
    );
    if (!sws) {
        if (err) *err = "sws_getContext failed";
        cleanup();
        return false;
    }

    while (av_read_frame(format, packet) >= 0) {
        if (packet->stream_index != stream_index) {
            av_packet_unref(packet);
            continue;
        }
        if (avcodec_send_packet(codec, packet) < 0) {
            av_packet_unref(packet);
            continue;
        }
        av_packet_unref(packet);

        if (avcodec_receive_frame(codec, frame) == 0) {
            if (av_frame_make_writable(rgb) < 0) {
                if (err) *err = "av_frame_make_writable failed";
                cleanup();
                return false;
            }

            sws_scale(
                sws,
                frame->data,
                frame->linesize,
                0,
                out_h,
                rgb->data,
                rgb->linesize
            );

            const int row_bytes = out_w * 3;
            out.resize((size_t)out_w * (size_t)out_h * 3ULL);
            for (int y = 0; y < out_h; ++y) {
                const uint8_t *row = rgb->data[0] + (size_t)y * (size_t)rgb->linesize[0];
                std::memcpy(out.data() + (size_t)y * (size_t)row_bytes, row, (size_t)row_bytes);
            }
            cleanup();
            return true;
        }
    }

    if (err) *err = "video decode produced no frame";
    cleanup();
    return false;
}

static inline bool ffmpegMediaHasAudioStream(const std::string &media_file)
{
    AVFormatContext *format = nullptr;
    if (avformat_open_input(&format, media_file.c_str(), nullptr, nullptr) < 0) return false;
    const bool has_audio = avformat_find_stream_info(format, nullptr) >= 0 &&
        av_find_best_stream(format, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0) >= 0;
    avformat_close_input(&format);
    return has_audio;
}

class FFmpegVideoFrameStream {
public:
    FFmpegVideoFrameStream(const std::string& video_file, int target_w, int target_h)
        : target_w_(target_w), target_h_(target_h)
    {
        packet_ = av_packet_alloc();
        frame_ = av_frame_alloc();
        if (!packet_ || !frame_ || target_w_ <= 0 || target_h_ <= 0) {
            error_ = "ffmpeg allocation or target dimensions invalid";
            return;
        }
        if (avformat_open_input(&format_, video_file.c_str(), nullptr, nullptr) < 0 ||
            avformat_find_stream_info(format_, nullptr) < 0) {
            error_ = "unable to open video stream";
            return;
        }
        stream_index_ = av_find_best_stream(format_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (stream_index_ < 0) {
            error_ = "no video stream found";
            return;
        }
        AVStream *stream = format_->streams[stream_index_];
        const AVCodec *decoder = avcodec_find_decoder(stream->codecpar->codec_id);
        if (!decoder) {
            error_ = "video decoder not found";
            return;
        }
        codec_ = avcodec_alloc_context3(decoder);
        if (!codec_ || avcodec_parameters_to_context(codec_, stream->codecpar) < 0 ||
            avcodec_open2(codec_, decoder, nullptr) < 0) {
            error_ = "video decoder initialization failed";
            return;
        }
        valid_ = true;
    }

    ~FFmpegVideoFrameStream()
    {
        if (sws_) sws_freeContext(sws_);
        if (codec_) avcodec_free_context(&codec_);
        if (format_) avformat_close_input(&format_);
        if (frame_) av_frame_free(&frame_);
        if (packet_) av_packet_free(&packet_);
    }

    FFmpegVideoFrameStream(const FFmpegVideoFrameStream&) = delete;
    FFmpegVideoFrameStream& operator=(const FFmpegVideoFrameStream&) = delete;

    bool valid() const { return valid_; }
    const std::string& error() const { return error_; }

    bool nextFrame(std::vector<uint8_t>& rgb)
    {
        rgb.clear();
        if (!valid_ || finished_) return false;

        while (true) {
            const int receive_result = avcodec_receive_frame(codec_, frame_);
            if (receive_result == 0) return convertCurrentFrame(rgb);
            if (receive_result == AVERROR_EOF) {
                finished_ = true;
                return false;
            }
            if (receive_result != AVERROR(EAGAIN)) {
                error_ = "video frame decode failed";
                finished_ = true;
                return false;
            }

            if (input_eof_) {
                const int flush_result = avcodec_send_packet(codec_, nullptr);
                if (flush_result < 0 && flush_result != AVERROR_EOF) {
                    error_ = "video decoder flush failed";
                    finished_ = true;
                    return false;
                }
                continue;
            }

            int read_result = 0;
            do {
                read_result = av_read_frame(format_, packet_);
                if (read_result < 0) {
                    input_eof_ = true;
                    break;
                }
                if (packet_->stream_index != stream_index_) av_packet_unref(packet_);
            } while (read_result >= 0 && packet_->stream_index != stream_index_);

            if (input_eof_) continue;
            const int send_result = avcodec_send_packet(codec_, packet_);
            av_packet_unref(packet_);
            if (send_result < 0 && send_result != AVERROR(EAGAIN)) {
                error_ = "video packet decode failed";
                finished_ = true;
                return false;
            }
        }
    }

private:
    bool convertCurrentFrame(std::vector<uint8_t>& rgb)
    {
        sws_ = sws_getCachedContext(
            sws_, frame_->width, frame_->height, static_cast<AVPixelFormat>(frame_->format),
            target_w_, target_h_, AV_PIX_FMT_RGB24, SWS_BICUBIC, nullptr, nullptr, nullptr);
        if (!sws_) {
            error_ = "video RGB conversion initialization failed";
            finished_ = true;
            return false;
        }
        rgb.resize(static_cast<size_t>(target_w_) * static_cast<size_t>(target_h_) * 3ULL);
        uint8_t *destination[] = {rgb.data(), nullptr, nullptr, nullptr};
        const int destination_linesize[] = {target_w_ * 3, 0, 0, 0};
        const int rows = sws_scale(
            sws_, frame_->data, frame_->linesize, 0, frame_->height,
            destination, destination_linesize);
        if (rows != target_h_) {
            rgb.clear();
            error_ = "video RGB conversion failed";
            finished_ = true;
            return false;
        }
        return true;
    }

    AVFormatContext *format_ = nullptr;
    AVCodecContext *codec_ = nullptr;
    SwsContext *sws_ = nullptr;
    AVPacket *packet_ = nullptr;
    AVFrame *frame_ = nullptr;
    int stream_index_ = -1;
    int target_w_ = 0;
    int target_h_ = 0;
    bool valid_ = false;
    bool input_eof_ = false;
    bool finished_ = false;
    std::string error_;
};

static inline bool ffmpegDecodeImageFramesToRgb(const std::string &image_file,
                                                std::vector<uint8_t> &frames,
                                                std::vector<int> &delays_ms,
                                                int &out_w,
                                                int &out_h,
                                                int &loop_count,
                                                size_t max_decoded_bytes,
                                                std::string *err = nullptr)
{
    frames.clear();
    delays_ms.clear();
    out_w = 0;
    out_h = 0;
    loop_count = 1;

    AVFormatContext *format = nullptr;
    AVCodecContext *codec = nullptr;
    SwsContext *sws = nullptr;
    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    if (!packet || !frame) {
        if (err) *err = "ffmpeg allocation failed";
        if (packet) av_packet_free(&packet);
        if (frame) av_frame_free(&frame);
        return false;
    }

    auto cleanup = [&]() {
        if (sws) sws_freeContext(sws);
        if (codec) avcodec_free_context(&codec);
        if (format) avformat_close_input(&format);
        av_frame_free(&frame);
        av_packet_free(&packet);
    };

    if (avformat_open_input(&format, image_file.c_str(), nullptr, nullptr) < 0 ||
        avformat_find_stream_info(format, nullptr) < 0) {
        if (err) *err = "ffmpeg cannot open image";
        cleanup();
        return false;
    }
    const int stream_index = av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (stream_index < 0) {
        if (err) *err = "ffmpeg image stream not found";
        cleanup();
        return false;
    }

    AVStream *stream = format->streams[stream_index];
    const AVCodec *decoder = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!decoder) {
        if (err) *err = "ffmpeg image decoder not found";
        cleanup();
        return false;
    }
    codec = avcodec_alloc_context3(decoder);
    if (!codec || avcodec_parameters_to_context(codec, stream->codecpar) < 0 ||
        avcodec_open2(codec, decoder, nullptr) < 0) {
        if (err) *err = "ffmpeg image decoder initialization failed";
        cleanup();
        return false;
    }

    if (const AVDictionaryEntry *entry = av_dict_get(stream->metadata, "loop", nullptr, 0)) {
        try { loop_count = std::stoi(entry->value); } catch (...) {}
    } else if (const AVDictionaryEntry *entry = av_dict_get(format->metadata, "loop", nullptr, 0)) {
        try { loop_count = std::stoi(entry->value); } catch (...) {}
    }

    std::vector<int64_t> timestamps;
    const AVRational frame_rate = av_guess_frame_rate(format, stream, nullptr);
    const int default_delay = frame_rate.num > 0 && frame_rate.den > 0
        ? std::max(1, static_cast<int>(std::lround(1000.0 * frame_rate.den / frame_rate.num)))
        : 100;

    auto receive_frames = [&]() -> bool {
        while (true) {
            const int receive_result = avcodec_receive_frame(codec, frame);
            if (receive_result == AVERROR(EAGAIN) || receive_result == AVERROR_EOF) return true;
            if (receive_result < 0) return false;

            if (frame->width <= 0 || frame->height <= 0) return false;
            if (out_w == 0 && out_h == 0) {
                out_w = frame->width;
                out_h = frame->height;
            } else if (frame->width != out_w || frame->height != out_h) {
                if (err) *err = "animated image changes canvas dimensions";
                return false;
            }

            const size_t frame_size = (size_t)out_w * (size_t)out_h * 3ULL;
            if (frame_size == 0 || frames.size() > max_decoded_bytes ||
                frame_size > max_decoded_bytes - frames.size()) {
                if (err) *err = "animated image exceeds dataset memory budget";
                return false;
            }

            sws = sws_getCachedContext(sws, out_w, out_h, static_cast<AVPixelFormat>(frame->format),
                                       out_w, out_h, AV_PIX_FMT_RGB24, SWS_BICUBIC,
                                       nullptr, nullptr, nullptr);
            if (!sws) return false;

            const size_t offset = frames.size();
            frames.resize(offset + frame_size);
            uint8_t *destination[4] = {frames.data() + offset, nullptr, nullptr, nullptr};
            int destination_stride[4] = {out_w * 3, 0, 0, 0};
            if (sws_scale(sws, frame->data, frame->linesize, 0, out_h,
                          destination, destination_stride) <= 0) {
                return false;
            }
            timestamps.push_back(frame->best_effort_timestamp);
            av_frame_unref(frame);
        }
    };

    bool decoded = true;
    while (av_read_frame(format, packet) >= 0) {
        if (packet->stream_index == stream_index) {
            const int send_result = avcodec_send_packet(codec, packet);
            if (send_result < 0 && send_result != AVERROR(EAGAIN)) decoded = false;
            if (decoded && !receive_frames()) decoded = false;
        }
        av_packet_unref(packet);
        if (!decoded) break;
    }
    if (decoded && avcodec_send_packet(codec, nullptr) >= 0) decoded = receive_frames();

    if (!decoded || frames.empty()) {
        if (err && err->empty()) *err = "ffmpeg image decode produced no frame";
        cleanup();
        frames.clear();
        return false;
    }

    delays_ms.assign(timestamps.size(), default_delay);
    for (size_t index = 0; index + 1 < timestamps.size(); ++index) {
        if (timestamps[index] == AV_NOPTS_VALUE || timestamps[index + 1] == AV_NOPTS_VALUE) continue;
        const double milliseconds = static_cast<double>(timestamps[index + 1] - timestamps[index]) *
                                    av_q2d(stream->time_base) * 1000.0;
        if (milliseconds > 0.0 && std::isfinite(milliseconds)) {
            delays_ms[index] = std::max(1, static_cast<int>(std::lround(milliseconds)));
        }
    }
    if (delays_ms.size() > 1) delays_ms.back() = delays_ms[delays_ms.size() - 2];

    cleanup();
    return true;
}

#endif

// NOTE: STB_IMAGE_IMPLEMENTATION is defined in src/stb_image_impl.cpp
// Do NOT define it here to avoid multiple definition errors
#include "stb_image.h"

// Lit tout le fichier en bytes. Retourne true si OK.
static inline bool readFileToVector(const std::string &path, std::vector<uint8_t> &out, std::string *err = nullptr)
{
    out.clear();
    try
    {
        std::ifstream ifs(path, std::ios::binary | std::ios::ate);
        if (!ifs)
        {
            if (err) *err = "open failed";
            return false;
        }
        std::streamsize sz = ifs.tellg();
        ifs.seekg(0, std::ios::beg);
        if (sz <= 0)
        {
            out.clear();
            return true;
        }
        out.resize(static_cast<size_t>(sz));
        if (!ifs.read(reinterpret_cast<char *>(out.data()), sz))
        {
            if (err) *err = "read failed";
            return false;
        }
        return true;
    }
    catch (const std::exception &e)
    {
        if (err) *err = e.what();
        return false;
    }
    catch (...)
    {
        if (err) *err = "unknown error";
        return false;
    }
}

static inline std::string bytesToString(const std::vector<uint8_t> &buf, size_t off = 0, size_t len = std::string::npos)
{
    if (off >= buf.size()) return std::string();
    size_t avail = buf.size() - off;
    if (len == std::string::npos || len > avail) len = avail;
    return std::string(reinterpret_cast<const char *>(buf.data() + off), len);
}

// remplacement de stb_image_resize par une fonction interne simple (nearest-neighbor)
static inline void resizeNearest(const unsigned char *src, int iw, int ih, int channels,
                                 unsigned char *dst, int ow, int oh)
{
    for (int y = 0; y < oh; ++y)
    {
        int sy = (y * ih) / oh;
        for (int x = 0; x < ow; ++x)
        {
            int sx = (x * iw) / ow;
            const unsigned char *s = src + (sy * iw + sx) * channels;
            unsigned char *d = dst + (y * ow + x) * channels;
            for (int c = 0; c < channels; ++c)
                d[c] = s[c];
        }
    }
}

// Redimensionnement bilinéaire (déterministe) pour images u8 interleavées.
// Utile surtout en downscale pour éviter les artefacts de nearest-neighbor.
static inline void resizeBilinear(const unsigned char* src, int iw, int ih, int channels,
                                  unsigned char* dst, int ow, int oh)
{
    if (!src || !dst || iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0 || channels <= 0) return;

    // Cas dégénérés: fallback nearest.
    if (iw == 1 || ih == 1 || ow == 1 || oh == 1) {
        resizeNearest(src, iw, ih, channels, dst, ow, oh);
        return;
    }

    const float sx = (float)iw / (float)ow;
    const float sy = (float)ih / (float)oh;

    for (int y = 0; y < oh; ++y) {
        const float fy = ((float)y + 0.5f) * sy - 0.5f;
        int y0 = (int)std::floor(fy);
        float wy = fy - (float)y0;
        if (y0 < 0) { y0 = 0; wy = 0.0f; }
        int y1 = y0 + 1;
        if (y1 >= ih) { y1 = ih - 1; }

        for (int x = 0; x < ow; ++x) {
            const float fx = ((float)x + 0.5f) * sx - 0.5f;
            int x0 = (int)std::floor(fx);
            float wx = fx - (float)x0;
            if (x0 < 0) { x0 = 0; wx = 0.0f; }
            int x1 = x0 + 1;
            if (x1 >= iw) { x1 = iw - 1; }

            const unsigned char* p00 = src + ((y0 * iw + x0) * channels);
            const unsigned char* p10 = src + ((y0 * iw + x1) * channels);
            const unsigned char* p01 = src + ((y1 * iw + x0) * channels);
            const unsigned char* p11 = src + ((y1 * iw + x1) * channels);

            unsigned char* out = dst + ((y * ow + x) * channels);

            const float w00 = (1.0f - wx) * (1.0f - wy);
            const float w10 = wx * (1.0f - wy);
            const float w01 = (1.0f - wx) * wy;
            const float w11 = wx * wy;

            for (int c = 0; c < channels; ++c) {
                const float v =
                    w00 * (float)p00[c] +
                    w10 * (float)p10[c] +
                    w01 * (float)p01[c] +
                    w11 * (float)p11[c];
                int iv = (int)std::lround(v);
                iv = std::clamp(iv, 0, 255);
                out[c] = (unsigned char)iv;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Resize haute qualité (bicubique). Pour RGB, on travaille en linéaire sRGB
// afin d'éviter les artefacts gamma visibles sur les gradients.
// ---------------------------------------------------------------------------

static inline float __srgb_u8_to_linear_f(unsigned char u) {
    const float s = (float)u / 255.0f;
    // sRGB exact piecewise
    if (s <= 0.04045f) return s / 12.92f;
    return std::pow((s + 0.055f) / 1.055f, 2.4f);
}

static inline unsigned char __linear_f_to_srgb_u8(float l) {
    l = std::clamp(l, 0.0f, 1.0f);
    float s = 0.0f;
    if (l <= 0.0031308f) s = 12.92f * l;
    else s = 1.055f * std::pow(l, 1.0f / 2.4f) - 0.055f;
    int v = (int)std::lround(s * 255.0f);
    v = std::clamp(v, 0, 255);
    return (unsigned char)v;
}

static inline float __cubic_catmull_rom(float x) {
    x = std::fabs(x);
    if (x <= 1.0f) {
        return (1.5f * x - 2.5f) * x * x + 1.0f;
    }
    if (x < 2.0f) {
        return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
    }
    return 0.0f;
}

// Bicubique générique sur u8 (sans correction gamma). Stable et déterministe.
static inline void resizeBicubicU8(const unsigned char* src, int iw, int ih, int channels,
                                   unsigned char* dst, int ow, int oh)
{
    if (!src || !dst || iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0 || channels <= 0) return;
    if (iw == ow && ih == oh) {
        std::memcpy(dst, src, (size_t)iw * (size_t)ih * (size_t)channels);
        return;
    }

    const float sx = (float)iw / (float)ow;
    const float sy = (float)ih / (float)oh;

    for (int y = 0; y < oh; ++y) {
        const float fy = ((float)y + 0.5f) * sy - 0.5f;
        const int iy = (int)std::floor(fy);

        float wy[4];
        for (int j = -1; j <= 2; ++j) {
            wy[j + 1] = __cubic_catmull_rom((float)(iy + j) - fy);
        }

        for (int x = 0; x < ow; ++x) {
            const float fx = ((float)x + 0.5f) * sx - 0.5f;
            const int ix = (int)std::floor(fx);

            float wx[4];
            for (int i = -1; i <= 2; ++i) {
                wx[i + 1] = __cubic_catmull_rom((float)(ix + i) - fx);
            }

            unsigned char* out = dst + ((y * ow + x) * channels);

            for (int c = 0; c < channels; ++c) {
                float acc = 0.0f;
                float wsum = 0.0f;

                for (int j = -1; j <= 2; ++j) {
                    int syi = iy + j;
                    syi = std::clamp(syi, 0, ih - 1);
                    const float wyj = wy[j + 1];
                    for (int i = -1; i <= 2; ++i) {
                        int sxi = ix + i;
                        sxi = std::clamp(sxi, 0, iw - 1);
                        const float w = wyj * wx[i + 1];
                        const unsigned char* p = src + ((syi * iw + sxi) * channels);
                        acc += w * (float)p[c];
                        wsum += w;
                    }
                }

                if (wsum != 0.0f) acc /= wsum;
                int iv = (int)std::lround(acc);
                iv = std::clamp(iv, 0, 255);
                out[c] = (unsigned char)iv;
            }
        }
    }
}

// Bicubique RGB en linéaire sRGB (meilleure qualité perçue, surtout en downscale).
static inline void resizeBicubicRGB_SRGBLinear(const unsigned char* src, int iw, int ih,
                                               unsigned char* dst, int ow, int oh)
{
    if (!src || !dst || iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0) return;
    if (iw == ow && ih == oh) {
        std::memcpy(dst, src, (size_t)iw * (size_t)ih * 3ULL);
        return;
    }

    // LUT sRGB->lin (256) pour accélérer.
    float lut[256];
    for (int i = 0; i < 256; ++i) lut[i] = __srgb_u8_to_linear_f((unsigned char)i);

    const float sx = (float)iw / (float)ow;
    const float sy = (float)ih / (float)oh;

    for (int y = 0; y < oh; ++y) {
        const float fy = ((float)y + 0.5f) * sy - 0.5f;
        const int iy = (int)std::floor(fy);

        float wy[4];
        for (int j = -1; j <= 2; ++j) {
            wy[j + 1] = __cubic_catmull_rom((float)(iy + j) - fy);
        }

        for (int x = 0; x < ow; ++x) {
            const float fx = ((float)x + 0.5f) * sx - 0.5f;
            const int ix = (int)std::floor(fx);

            float wx[4];
            for (int i = -1; i <= 2; ++i) {
                wx[i + 1] = __cubic_catmull_rom((float)(ix + i) - fx);
            }

            unsigned char* out = dst + ((y * ow + x) * 3);
            for (int c = 0; c < 3; ++c) {
                float acc = 0.0f;
                float wsum = 0.0f;
                for (int j = -1; j <= 2; ++j) {
                    int syi = std::clamp(iy + j, 0, ih - 1);
                    const float wyj = wy[j + 1];
                    for (int i = -1; i <= 2; ++i) {
                        int sxi = std::clamp(ix + i, 0, iw - 1);
                        const float w = wyj * wx[i + 1];
                        const unsigned char* p = src + ((syi * iw + sxi) * 3);
                        acc += w * lut[p[c]];
                        wsum += w;
                    }
                }
                if (wsum != 0.0f) acc /= wsum;
                out[c] = __linear_f_to_srgb_u8(acc);
            }
        }
    }
}

enum Modality : unsigned
{
    MOD_NONE = 0,
    MOD_TEXT = 1 << 0,
    MOD_AUDIO = 1 << 1,
    MOD_IMAGE = 1 << 2,
    MOD_VIDEO = 1 << 3
};

static inline unsigned detectModalities(const std::string &dataset_dir)
{
    static const std::regex re_text(R"(\.(txt|json|csv|md|xml)$)", std::regex::icase);
    static const std::regex re_audio(R"(\.(wav|flac|mp3|ogg|m4a|aac)$)", std::regex::icase);
    static const std::regex re_image(R"(\.(png|jpg|jpeg|gif|bmp|tiff|webp|ppm|pgm)$)", std::regex::icase);
    static const std::regex re_video(R"(\.(mp4|mkv|avi|mov|webm|flv|ts)$)", std::regex::icase);

    unsigned mask = MOD_NONE;
    for (auto &p : fs::recursive_directory_iterator(dataset_dir))
    {
        if (!p.is_regular_file()) continue;
        const std::string s = p.path().string();
        if (std::regex_search(s, re_text)) mask |= MOD_TEXT;
        if (std::regex_search(s, re_audio)) mask |= MOD_AUDIO;
        if (std::regex_search(s, re_image)) mask |= MOD_IMAGE;
        if (std::regex_search(s, re_video)) mask |= MOD_VIDEO;
        if (mask == (MOD_TEXT | MOD_AUDIO | MOD_IMAGE | MOD_VIDEO)) break;
    }
    return mask;
}

struct MagicToken
{
    uint32_t modality_mask; // bits Modality
    uint32_t seed;          // déterministe d'après dataset
    float embed[8];         // petit embedding fixe (8 dim)
};

// Hash SHA256 du dataset basé sur les noms et tailles de fichiers
static inline std::string datasetSHA256Hash(const std::string &dir, const std::string &excluded_cache_path = "")
{
    // Versionne la sémantique d'indexation afin de ne pas restaurer un cache
    // créé avant la prise en charge des sidecars associés à des symlinks.
    const fs::path lexical_root = fs::absolute(dir).lexically_normal();
    const fs::path excluded = excluded_cache_path.empty() ? fs::path() : fs::absolute(excluded_cache_path).lexically_normal();
    std::string combined = "mimir_dataset_index_v4_structured_annotations\n" + lexical_root.generic_string() + "\n";
    std::vector<std::string> file_infos;
    
    // Collecter tous les fichiers avec leurs métadonnées
    for (auto &p : fs::recursive_directory_iterator(dir)) {
        if (!p.is_regular_file()) continue;
        
        const auto lexical_path = fs::absolute(p.path()).lexically_normal();
        if (isDatasetInternalFile(p.path()) || lexical_path == excluded) continue;
        std::string info = lexical_path.lexically_relative(lexical_root).generic_string() + "|" +
                          std::to_string(fs::file_size(p.path())) + "|" +
                          std::to_string(fs::last_write_time(p.path()).time_since_epoch().count());
        // JSON edits of equal length must invalidate annotation selection too.
        std::string ext = p.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        if (ext == ".json") {
            std::ifstream input(p.path(), std::ios::binary);
            if (!input) throw std::runtime_error("Cannot hash dataset JSON: " + p.path().string());
            std::ostringstream content; content << input.rdbuf();
            info += "|" + sha256(content.str());
        }
        file_infos.push_back(info);
    }
    
    // Trier pour garantir un ordre déterministe
    std::sort(file_infos.begin(), file_infos.end());
    
    // Combiner toutes les infos
    for (const auto &info : file_infos) {
        combined += info + "\n";
    }
    
    // Calculer le SHA256
    return sha256(combined);
}

static inline uint32_t simpleDatasetHash(const std::string &dir)
{
    uint64_t h = 1469598103934665603ull; // FNV offset basis
    for (auto &p : fs::recursive_directory_iterator(dir))
    {
        if (!p.is_regular_file()) continue;
        auto s = p.path().filename().string();
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    }
    return static_cast<uint32_t>((h ^ (h >> 32)) & 0xFFFFFFFFu);
}

static inline MagicToken makeMagicToken(unsigned modality_mask, const std::string &dir)
{
    MagicToken t{};
    t.modality_mask = modality_mask;
    t.seed = simpleDatasetHash(dir) ^ (modality_mask * 0x9e3779b1u);
    std::mt19937 rng(t.seed);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    for (auto &e : t.embed) e = U(rng);
    return t;
}

static inline std::string sanitize_utf8(const std::string &s)
{
    const unsigned char *b = reinterpret_cast<const unsigned char *>(s.data());
    size_t n = s.size();
    std::string out;
    out.reserve(n);
    size_t i = 0;
    while (i < n)
    {
        unsigned char c = b[i];
        if (c < 0x80) { out.push_back((char)c); ++i; continue; }
        int len = 0;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        else { out.append("\xEF\xBF\xBD"); ++i; continue; } // U+FFFD
        if (i + (size_t)len > n) { out.append("\xEF\xBF\xBD"); ++i; continue; }
        bool ok = true;
        for (int k = 1; k < len; ++k) if ((b[i + k] & 0xC0) != 0x80) { ok = false; break; }
        if (!ok) { out.append("\xEF\xBF\xBD"); ++i; continue; }
        for (int k = 0; k < len; ++k) out.push_back((char)b[i + k]);
        i += (size_t)len;
    }
    return out;
}

// image -> embedding helper (static inline)
static inline std::vector<float> imageToEmbedding(const std::vector<uint8_t> &img, int iw, int ih, int dim)
{
    std::vector<float> out((size_t)dim, 0.0f);
    if (img.empty() || iw <= 0 || ih <= 0 || dim <= 0) return out;
    uint64_t sum = 0;
    size_t N = (size_t)iw * (size_t)ih;
    size_t limit = std::min(N, img.size());
    for (size_t i = 0; i < limit; ++i) sum += img[i];
    double mean = (limit > 0) ? (double(sum) / double(limit)) : 0.0;
    float v = static_cast<float>((mean / 255.0) * 2.0 - 1.0f);
    for (int d = 0; d < dim; ++d) out[(size_t)d] = v;
    return out;
}

// helper: write uint32 little-endian (used by packToSafetensor)
static inline void write_u32_le(std::ofstream &f, uint32_t v)
{
    unsigned char b[4];
    b[0] = static_cast<unsigned char>(v & 0xFF);
    b[1] = static_cast<unsigned char>((v >> 8) & 0xFF);
    b[2] = static_cast<unsigned char>((v >> 16) & 0xFF);
    b[3] = static_cast<unsigned char>((v >> 24) & 0xFF);
    f.write(reinterpret_cast<char *>(b), 4);
}

// Gestionnaire global de mémoire RAM pour le dataset
struct DatasetMemoryManager {
    static DatasetMemoryManager& instance() {
        // Leaky singleton: évite les crashs d'ordre de destruction des statics
        // quand des DatasetItem sont détruits pendant l'atexit.
        static DatasetMemoryManager* mgr = new DatasetMemoryManager();
        return *mgr;
    }
    
    size_t max_ram_bytes = 10ULL * 1024 * 1024 * 1024; // 10 GB par défaut
    size_t current_ram_bytes = 0;
    size_t peak_ram_bytes = 0;
    std::unordered_map<void*, size_t> allocations; // Tracker les allocations
    
    void setMaxRAM(size_t bytes) { max_ram_bytes = bytes; }
    
    bool canAllocate(size_t bytes) const {
        return (current_ram_bytes + bytes) <= max_ram_bytes;
    }
    
    void trackAllocation(void* ptr, size_t bytes) {
        current_ram_bytes += bytes;
        peak_ram_bytes = std::max(peak_ram_bytes, current_ram_bytes);
        allocations[ptr] = bytes;
    }
    
    void trackDeallocation(void* ptr) {
        auto it = allocations.find(ptr);
        if (it != allocations.end()) {
            current_ram_bytes -= it->second;
            allocations.erase(it);
        }
    }
    
    size_t getCurrentRAM() const { return current_ram_bytes; }
    size_t getPeakRAM() const { return peak_ram_bytes; }
    size_t getAvailableRAM() const { 
        return max_ram_bytes > current_ram_bytes ? max_ram_bytes - current_ram_bytes : 0; 
    }
    
    float getUsagePercent() const {
        return 100.0f * static_cast<float>(current_ram_bytes) / static_cast<float>(max_ram_bytes);
    }
    
    void printStats() const {
        std::cerr << "\n💾 RAM Manager Stats:" << std::endl;
        std::cerr << "   Current: " << (current_ram_bytes / 1024 / 1024) << " MB" << std::endl;
        std::cerr << "   Peak:    " << (peak_ram_bytes / 1024 / 1024) << " MB" << std::endl;
        std::cerr << "   Max:     " << (max_ram_bytes / 1024 / 1024) << " MB" << std::endl;
        std::cerr << "   Usage:   " << std::fixed << std::setprecision(1) 
                  << getUsagePercent() << "%" << std::endl;
        std::cerr << "   Active allocations: " << allocations.size() << std::endl;
    }
    
private:
    DatasetMemoryManager() = default;
};

// Structure d'élément de dataset avec gestion intelligente de la RAM
struct DatasetItem {
    std::string name;
    bool is_linked = false;
    
    // Chemins (toujours présents - négligeable en RAM)
    std::string text_file;
    std::string text_inline;
    std::string annotation_file; // Structured sidecar; also used as text_file for existing trainers.
    std::optional<json> annotation; // Lazy, intact metadata (normalized bboxes, relations, etc.).
    std::string annotation_error;
    std::string image_file;
    std::string mask_file;
    std::string audio_file;
    std::string video_file;
    
    // Données (chargées à la demande avec gestion RAM)
    std::optional<std::string> text;
    std::vector<uint8_t> img;
    std::vector<uint8_t> animation_frames;
    std::vector<int> animation_frame_delays_ms;
    int animation_loop_count = 1; // 0 = boucle infinie
    std::vector<uint8_t> mask;
    std::vector<uint8_t> audio_bytes;
    std::vector<uint8_t> video_bytes;

    bool img_loaded = false;
    bool mask_loaded = false;
    bool audio_loaded = false;
    bool video_loaded = false;
    
    // Métadonnées
    int w = 0, h = 0;
    int img_c = 0; // 0 = inconnu, 1 = grayscale, 3 = RGB
    
    // Tracking RAM et LRU
    mutable uint64_t last_access_time = 0;
    mutable size_t estimated_ram_usage = 0;

    ~DatasetItem();
    
    // Libère toutes les données chargées
    void unload() {
        auto& mgr = DatasetMemoryManager::instance();
        
        if (text.has_value()) {
            mgr.trackDeallocation((void*)text->data());
            text.reset();
        }
        if (annotation.has_value()) {
            mgr.trackDeallocation((void*)&*annotation);
            annotation.reset();
        }
        annotation_error.clear();
        if (img_loaded) {
            void* ptr = img.empty() ? nullptr : (void*)img.data();
            if (ptr) mgr.trackDeallocation(ptr);
            std::vector<uint8_t>().swap(img);
            img_loaded = false;
        }
        if (!animation_frames.empty()) {
            mgr.trackDeallocation(animation_frames.data());
            std::vector<uint8_t>().swap(animation_frames);
        }
        std::vector<int>().swap(animation_frame_delays_ms);
        animation_loop_count = 1;
        if (mask_loaded) {
            void* ptr = mask.empty() ? nullptr : (void*)mask.data();
            if (ptr) mgr.trackDeallocation(ptr);
            std::vector<uint8_t>().swap(mask);
            mask_loaded = false;
        }
        if (audio_loaded) {
            void* ptr = audio_bytes.empty() ? nullptr : (void*)audio_bytes.data();
            if (ptr) mgr.trackDeallocation(ptr);
            std::vector<uint8_t>().swap(audio_bytes);
            audio_loaded = false;
        }
        if (video_loaded) {
            void* ptr = video_bytes.empty() ? nullptr : (void*)video_bytes.data();
            if (ptr) mgr.trackDeallocation(ptr);
            std::vector<uint8_t>().swap(video_bytes);
            video_loaded = false;
        }
        
        estimated_ram_usage = 0;
    }
    
    // Estime la RAM nécessaire pour charger un item
    size_t estimateRAMNeeded() const {
        size_t total = 0;
        
        if ((!text_file.empty() || !text_inline.empty()) && !text.has_value()) {
            try {
                if (!text_inline.empty()) total += text_inline.size() * 2;
                else {
                    auto fsize = fs::file_size(text_file);
                    total += fsize * 2; // UTF-8 peut doubler
                }
            } catch (...) {}
        }
        
        if (!image_file.empty() && !img_loaded) {
            // Estimation basée sur la taille cible
            const size_t c = 3; // Les images sont chargées en RGB.
            total += (size_t)w * (size_t)h * c;
        }

        if (!mask_file.empty() && !mask_loaded) {
            total += (size_t)w * (size_t)h;
        }
        
        if (!audio_file.empty() && !audio_loaded) {
            try {
                total += fs::file_size(audio_file);
            } catch (...) {}
        }
        
        if (!video_file.empty() && !video_loaded) {
            try {
                total += fs::file_size(video_file);
            } catch (...) {}
        }
        
        return total;
    }
    
    // Met à jour le timestamp d'accès (pour LRU)
    void touch() const {
        static uint64_t global_counter = 0;
        last_access_time = ++global_counter;
    }
    
    // Lazy loaders avec gestion RAM intelligente
    bool loadText() {
        if (text.has_value()) {
            touch();
            return true;
        }
        if (text_file.empty() && text_inline.empty()) return false;
        annotation_error.clear();

        auto& mgr = DatasetMemoryManager::instance();
        
        try {
            // Estimer la taille
            size_t file_size = !text_inline.empty() ? text_inline.size() : fs::file_size(text_file);
            size_t needed = file_size * 2; // Sécurité UTF-8
            
            // Vérifier si on peut allouer
            if (!mgr.canAllocate(needed)) {
                return false; // Pas assez de RAM
            }
            
            std::string loaded;
            std::optional<json> loaded_annotation;
            size_t annotation_bytes = 0;
            if (!annotation_file.empty()) {
                std::ifstream f(annotation_file);
                if (!f) throw std::runtime_error("Cannot open annotation: " + annotation_file);
                json document; f >> document;
                loaded = sanitize_utf8(datasetAnnotationCaption(document, image_file));
                annotation_bytes = document.dump().size() + sizeof(json);
                if (!mgr.canAllocate(annotation_bytes + loaded.size())) return false;
                loaded_annotation = std::move(document);
            } else if (!text_inline.empty()) {
                loaded = sanitize_utf8(text_inline);
            } else {
                std::ifstream f(text_file);
                if (!f) return false;
                std::ostringstream ss;
                ss << f.rdbuf();
                loaded = sanitize_utf8(ss.str());
            }
            size_t actual_size = loaded.size();
            
            if (loaded_annotation) {
                annotation = std::move(loaded_annotation);
                mgr.trackAllocation((void*)&*annotation, annotation_bytes);
                estimated_ram_usage += annotation_bytes;
            }
            text = std::move(loaded);
            mgr.trackAllocation((void*)text->data(), actual_size);
            estimated_ram_usage += actual_size;
            touch();
            
            return true;
        } catch (const std::exception& e) {
            if (!annotation_file.empty()) annotation_error = annotation_file + ": " + e.what();
            return false;
        }
    }
    
    // Le chargement générique conserve la couleur comme les chemins vision.
    bool loadImage(int target_w, int target_h) {
        return loadImageRGB(target_w, target_h);
    }

    // Charge en RGB et redimensionne par interpolation bicubique en linéaire sRGB.
    bool loadImageRGB(int target_w, int target_h) {
        if (target_w <= 0 || target_h <= 0) return false;
        if (img_loaded && img_c == 3 && w == target_w && h == target_h) {
            touch();
            return true;
        }
        if (image_file.empty()) return false;

        auto& mgr = DatasetMemoryManager::instance();

        try {
            // Estimer la RAM nécessaire
            size_t needed = (size_t)target_w * (size_t)target_h * 3 * 2; // temporaire + final
            if (!mgr.canAllocate(needed)) {
                return false;
            }

            std::string extension = fs::path(image_file).extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });

            int w_img = 0, h_img = 0, c = 0;
            int frame_count = 1;
            std::vector<unsigned char> source_frames;
            std::vector<int> frame_delays;
            int loop_count = 1;

            if (extension == ".gif") {
                std::vector<uint8_t> encoded;
                if (!readFileToVector(image_file, encoded) || encoded.empty() ||
                    encoded.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    return false;
                }

                int* delays = nullptr;
                unsigned char* rgba = stbi_load_gif_from_memory(
                    encoded.data(), static_cast<int>(encoded.size()), &delays,
                    &w_img, &h_img, &frame_count, &c, 4);
                if (!rgba || w_img <= 0 || h_img <= 0 || frame_count <= 0) {
                    if (rgba) stbi_image_free(rgba);
                    if (delays) stbi_image_free(delays);
                    return false;
                }

                const size_t source_pixels = (size_t)w_img * (size_t)h_img;
                source_frames.resize(source_pixels * (size_t)frame_count * 3ULL);
                for (size_t pixel = 0; pixel < source_pixels * (size_t)frame_count; ++pixel) {
                    const unsigned alpha = rgba[pixel * 4ULL + 3ULL];
                    for (size_t channel = 0; channel < 3; ++channel) {
                        source_frames[pixel * 3ULL + channel] = static_cast<uint8_t>(
                            (static_cast<unsigned>(rgba[pixel * 4ULL + channel]) * alpha + 127U) / 255U);
                    }
                }
                if (delays) frame_delays.assign(delays, delays + frame_count);
                else frame_delays.assign((size_t)frame_count, 100);
                stbi_image_free(rgba);
                if (delays) stbi_image_free(delays);

                static const unsigned char application_id[] = {
                    'N','E','T','S','C','A','P','E','2','.','0'
                };
                const auto loop_pos = std::search(encoded.begin(), encoded.end(),
                                                  std::begin(application_id), std::end(application_id));
                if (loop_pos != encoded.end()) {
                    const size_t offset = static_cast<size_t>(loop_pos - encoded.begin()) + sizeof(application_id);
                    if (offset + 4 < encoded.size() && encoded[offset] == 3 && encoded[offset + 1] == 1) {
                        loop_count = static_cast<int>(encoded[offset + 2]) |
                                     (static_cast<int>(encoded[offset + 3]) << 8);
                    }
                }
            }
#ifdef MIMIR_ENABLE_FFMPEG
            else if (extension == ".webp") {
                std::string decode_error;
                if (!ffmpegDecodeImageFramesToRgb(image_file, source_frames, frame_delays,
                                                  w_img, h_img, loop_count,
                                                  mgr.getAvailableRAM(), &decode_error)) {
                    return false;
                }
                const size_t source_frame_size = (size_t)w_img * (size_t)h_img * 3ULL;
                if (source_frame_size == 0 || source_frames.size() % source_frame_size != 0) return false;
                frame_count = static_cast<int>(source_frames.size() / source_frame_size);
            }
#endif
            else {
                unsigned char* data = stbi_load(image_file.c_str(), &w_img, &h_img, &c, 3);
                if (data && w_img > 0 && h_img > 0) {
                    source_frames.resize((size_t)w_img * (size_t)h_img * 3ULL);
                    std::memcpy(source_frames.data(), data, source_frames.size());
                    stbi_image_free(data);
                } else {
                    if (data) stbi_image_free(data);
#ifdef MIMIR_ENABLE_FFMPEG
                    if (extension == ".bmp") {
                        std::string decode_error;
                        if (!ffmpegDecodeImageFramesToRgb(image_file, source_frames, frame_delays,
                                                          w_img, h_img, loop_count,
                                                          mgr.getAvailableRAM(), &decode_error)) {
                            return false;
                        }
                        const size_t source_frame_size = (size_t)w_img * (size_t)h_img * 3ULL;
                        if (source_frame_size == 0 || source_frames.size() % source_frame_size != 0) return false;
                        frame_count = static_cast<int>(source_frames.size() / source_frame_size);
                    } else {
                        return false;
                    }
#else
                    return false;
#endif
                }
            }

            const size_t target_frame_size = (size_t)target_w * (size_t)target_h * 3ULL;
            std::vector<uint8_t> resized_frames(target_frame_size * (size_t)frame_count);
            const size_t source_frame_size = (size_t)w_img * (size_t)h_img * 3ULL;
            for (int frame = 0; frame < frame_count; ++frame) {
                resizeBicubicRGB_SRGBLinear(
                    source_frames.data() + (size_t)frame * source_frame_size, w_img, h_img,
                    resized_frames.data() + (size_t)frame * target_frame_size, target_w, target_h);
            }
            Mimir::fpgaStageDataset(resized_frames);
            std::vector<uint8_t> dst(resized_frames.begin(), resized_frames.begin() + target_frame_size);

            const size_t persistent_size = dst.size() + (frame_count > 1 ? resized_frames.size() : 0ULL);
            if (!mgr.canAllocate(persistent_size)) return false;

            // Si on avait déjà une image chargée, décrémenter l'ancien tracking
            if (img_loaded) {
                void* ptr = img.empty() ? nullptr : (void*)img.data();
                if (ptr) mgr.trackDeallocation(ptr);
                if (estimated_ram_usage >= img.size()) estimated_ram_usage -= img.size();
            }
            if (!animation_frames.empty()) {
                mgr.trackDeallocation(animation_frames.data());
                if (estimated_ram_usage >= animation_frames.size()) estimated_ram_usage -= animation_frames.size();
                std::vector<uint8_t>().swap(animation_frames);
            }

            const size_t actual_size = dst.size();
            img = std::move(dst);
            img_loaded = true;
            mgr.trackAllocation((void*)img.data(), actual_size);
            estimated_ram_usage += actual_size;
            if (frame_count > 1) {
                animation_frames = std::move(resized_frames);
                animation_frame_delays_ms = std::move(frame_delays);
                animation_loop_count = loop_count;
                mgr.trackAllocation(animation_frames.data(), animation_frames.size());
                estimated_ram_usage += animation_frames.size();
            } else {
                animation_frame_delays_ms.clear();
                animation_loop_count = 1;
            }

            w = target_w;
            h = target_h;
            img_c = 3;
            touch();
            return true;
        } catch (...) {
            return false;
        }
    }

    size_t animationFrameCount() const {
        if (!animation_frames.empty() && w > 0 && h > 0 && img_c > 0) {
            const size_t frame_size = (size_t)w * (size_t)h * (size_t)img_c;
            return frame_size > 0 ? animation_frames.size() / frame_size : 0;
        }
        return img_loaded ? 1 : 0;
    }

    bool isAnimatedImage() const {
        return animationFrameCount() > 1;
    }

    // Charge l'annotation spatiale en un canal. Nearest-neighbor préserve les
    // classes discrètes, contrairement aux interpolations destinées aux photos.
    bool loadMask(int target_w, int target_h) {
        if (target_w <= 0 || target_h <= 0) return false;
        if (mask_loaded && w == target_w && h == target_h) {
            touch();
            return true;
        }
        if (mask_file.empty()) return false;

        auto& mgr = DatasetMemoryManager::instance();
        const size_t needed = (size_t)target_w * (size_t)target_h;
        if (!mgr.canAllocate(needed)) return false;

        int source_w = 0, source_h = 0, source_c = 0;
        unsigned char* source = stbi_load(mask_file.c_str(), &source_w, &source_h, &source_c, 1);
        if (!source || source_w <= 0 || source_h <= 0) {
            if (source) stbi_image_free(source);
            return false;
        }

        std::vector<uint8_t> resized(needed);
        for (int y = 0; y < target_h; ++y) {
            const int source_y = std::min(source_h - 1, y * source_h / target_h);
            for (int x = 0; x < target_w; ++x) {
                const int source_x = std::min(source_w - 1, x * source_w / target_w);
                resized[(size_t)y * (size_t)target_w + (size_t)x] =
                    source[(size_t)source_y * (size_t)source_w + (size_t)source_x];
            }
        }
        stbi_image_free(source);

        if (mask_loaded && !mask.empty()) {
            mgr.trackDeallocation(mask.data());
            if (estimated_ram_usage >= mask.size()) estimated_ram_usage -= mask.size();
        }
        mask = std::move(resized);
        mask_loaded = true;
        mgr.trackAllocation(mask.data(), mask.size());
        estimated_ram_usage += mask.size();
        w = target_w;
        h = target_h;
        touch();
        return true;
    }

    // Produit des poids HWC de moyenne 1: le contexte reste appris, tandis que
    // les pixels annotés contribuent davantage à la reconstruction.
    std::vector<float> makeMaskLossWeights(int channels = 3,
                                           float foreground_weight = 4.0f,
                                           float background_weight = 1.0f) const {
        if (!mask_loaded || mask.empty() || channels <= 0) return {};
        foreground_weight = std::max(0.0f, foreground_weight);
        background_weight = std::max(0.0f, background_weight);

        std::vector<float> weights(mask.size() * (size_t)channels);
        double sum = 0.0;
        for (size_t pixel = 0; pixel < mask.size(); ++pixel) {
            const float weight = mask[pixel] == 0 ? background_weight : foreground_weight;
            sum += weight;
            for (int channel = 0; channel < channels; ++channel) {
                weights[pixel * (size_t)channels + (size_t)channel] = weight;
            }
        }
        if (sum <= 0.0) return {};
        const float normalization = static_cast<float>((double)mask.size() / sum);
        for (float& weight : weights) weight *= normalization;
        return weights;
    }
    
    bool loadAudio() {
        if (audio_loaded) {
            touch();
            return true;
        }
        if (audio_file.empty()) return false;

#ifdef MIMIR_ENABLE_FFMPEG
        {
            std::vector<uint8_t> decoded_audio;
            std::string ffmpeg_err;
            if (ffmpegDecodeAudioToPcm(audio_file, decoded_audio, &ffmpeg_err)) {
                auto& mgr = DatasetMemoryManager::instance();
                if (!mgr.canAllocate(decoded_audio.size())) {
                    return false;
                }
                audio_bytes = std::move(decoded_audio);
                audio_loaded = true;
                mgr.trackAllocation((void*)audio_bytes.data(), audio_bytes.size());
                estimated_ram_usage += audio_bytes.size();
                touch();
                return true;
            }
        }
#endif
        
        auto& mgr = DatasetMemoryManager::instance();
        
        try {
            size_t file_size = fs::file_size(audio_file);
            
            if (!mgr.canAllocate(file_size)) {
                return false;
            }
            
            std::ifstream f(audio_file, std::ios::binary);
            if (!f) return false;
            std::vector<uint8_t> data(
                (std::istreambuf_iterator<char>(f)),
                std::istreambuf_iterator<char>()
            );
            
            size_t actual_size = data.size();
            audio_bytes = std::move(data);
            audio_loaded = true;
            mgr.trackAllocation((void*)audio_bytes.data(), actual_size);
            estimated_ram_usage += actual_size;
            touch();
            
            return true;
        } catch (...) {
            return false;
        }
    }
    
    bool loadVideo() {
        if (video_loaded) {
            touch();
            return true;
        }
        if (video_file.empty()) return false;

#ifdef MIMIR_ENABLE_FFMPEG
        {
            std::vector<uint8_t> decoded_video;
            int decoded_w = 0;
            int decoded_h = 0;
            std::string ffmpeg_err;
            if (ffmpegDecodeVideoToRgbFirstFrame(video_file, decoded_video, decoded_w, decoded_h, &ffmpeg_err)) {
                auto& mgr = DatasetMemoryManager::instance();
                if (!mgr.canAllocate(decoded_video.size())) {
                    return false;
                }

                video_bytes = std::move(decoded_video);
                video_loaded = true;
                mgr.trackAllocation((void*)video_bytes.data(), video_bytes.size());
                estimated_ram_usage += video_bytes.size();
                w = decoded_w;
                h = decoded_h;
                img_c = 3;
                touch();
                return true;
            }
        }
#endif
        
        auto& mgr = DatasetMemoryManager::instance();
        
        try {
            size_t file_size = fs::file_size(video_file);
            
            if (!mgr.canAllocate(file_size)) {
                return false;
            }
            
            std::ifstream f(video_file, std::ios::binary);
            if (!f) return false;
            std::vector<uint8_t> data(
                (std::istreambuf_iterator<char>(f)),
                std::istreambuf_iterator<char>()
            );
            
            size_t actual_size = data.size();
            video_bytes = std::move(data);
            video_loaded = true;
            mgr.trackAllocation((void*)video_bytes.data(), actual_size);
            estimated_ram_usage += actual_size;
            touch();
            
            return true;
        } catch (...) {
            return false;
        }
    }
    
    // Helper: compte les modalités disponibles
    int countModalities() const {
        int count = 0;
        if (!text_file.empty() || !text_inline.empty()) count++;
        if (!image_file.empty()) count++;
        if (!audio_file.empty()) count++;
        if (!video_file.empty()) count++;
        return count;
    }
    
    // Helper: complétude en pourcentage (0-100)
    float completeness() const {
        return (countModalities() / 4.0f) * 100.0f;
    }
    
    // Helper: taille RAM actuelle
    size_t getRAMUsage() const {
        return estimated_ram_usage;
    }
    
    // Helper: vérifie si des données sont chargées
    bool isLoaded() const {
        return text.has_value() || img_loaded || mask_loaded || audio_loaded || video_loaded;
    }
};

class DatasetItemUnloadGuard {
public:
    explicit DatasetItemUnloadGuard(DatasetItem& item) : item_(item) {}
    ~DatasetItemUnloadGuard() { item_.unload(); }

    DatasetItemUnloadGuard(const DatasetItemUnloadGuard&) = delete;
    DatasetItemUnloadGuard& operator=(const DatasetItemUnloadGuard&) = delete;

private:
    DatasetItem& item_;
};

// Gestionnaire de Dataset avec éviction LRU intelligente
class DatasetManager {
public:
    DatasetManager(size_t max_ram_mb = 10240) {
        auto& mgr = DatasetMemoryManager::instance();
        mgr.setMaxRAM(max_ram_mb * 1024ULL * 1024ULL);
    }
    
    // Charge les items nécessaires en libérant les plus anciens si besoin
    bool ensureLoaded(std::vector<DatasetItem>& items, const std::vector<size_t>& indices, 
                      int target_w = 64, int target_h = 64) {
        auto& mgr = DatasetMemoryManager::instance();
        
        // Calculer la RAM nécessaire
        size_t needed = 0;
        for (size_t idx : indices) {
            if (idx >= items.size()) continue;
            if (!items[idx].isLoaded()) {
                needed += items[idx].estimateRAMNeeded();
            }
        }
        
        // Si pas assez de RAM, évincer les items les moins récemment utilisés
        if (!mgr.canAllocate(needed)) {
            evictLRU(items, needed);
        }
        
        // Charger les items demandés
        bool all_loaded = true;
        for (size_t idx : indices) {
            if (idx >= items.size()) continue;
            
            auto& item = items[idx];
            
            // Charger selon les modalités disponibles
            if (!item.image_file.empty() && !item.img_loaded) {
                if (!item.loadImage(target_w, target_h)) {
                    all_loaded = false;
                }
            }

            if (!item.mask_file.empty() && !item.mask_loaded) {
                if (!item.loadMask(target_w, target_h)) {
                    all_loaded = false;
                }
            }
            
            if ((!item.text_file.empty() || !item.text_inline.empty()) && !item.text.has_value()) {
                if (!item.loadText()) {
                    all_loaded = false;
                }
            }

            if (!item.audio_file.empty() && !item.audio_loaded) {
                if (!item.loadAudio()) {
                    all_loaded = false;
                }
            }

            if (!item.video_file.empty() && !item.video_loaded) {
                if (!item.loadVideo()) {
                    all_loaded = false;
                }
            }
        }
        
        return all_loaded;
    }
    
    // Éviction LRU : libère les items les moins récemment utilisés
    void evictLRU(std::vector<DatasetItem>& items, size_t bytes_needed) {
        auto& mgr = DatasetMemoryManager::instance();
        
        // Créer une liste des items chargés triés par last_access_time
        std::vector<size_t> loaded_indices;
        for (size_t i = 0; i < items.size(); ++i) {
            if (items[i].isLoaded()) {
                loaded_indices.push_back(i);
            }
        }
        
        // Trier par access time (les plus anciens en premier)
        std::sort(loaded_indices.begin(), loaded_indices.end(), 
                  [&items](size_t a, size_t b) {
                      return items[a].last_access_time < items[b].last_access_time;
                  });
        
        // Évincer jusqu'à avoir assez de RAM
        size_t freed = 0;
        for (size_t idx : loaded_indices) {
            if (mgr.canAllocate(bytes_needed)) break;
            
            size_t item_size = items[idx].getRAMUsage();
            items[idx].unload();
            freed += item_size;
        }
        
        if (freed > 0) {
            std::cerr << "  ⟳ LRU éviction: libéré " << (freed / 1024 / 1024) << " MB" << std::endl;
        }
    }
    
    // Préchargement intelligent d'un batch
    bool preloadBatch(std::vector<DatasetItem>& items, size_t batch_start, size_t batch_size,
                     int target_w = 64, int target_h = 64) {
        std::vector<size_t> indices;
        for (size_t i = 0; i < batch_size && (batch_start + i) < items.size(); ++i) {
            indices.push_back(batch_start + i);
        }
        return ensureLoaded(items, indices, target_w, target_h);
    }
    
    // Statistiques
    void printStats() const {
        DatasetMemoryManager::instance().printStats();
    }
};

// implémentation de loadDataset (parcours récursif, indexation des linkables)
// Version optimisée RAM : ne charge PAS les données, seulement les métadonnées
static inline bool loadCocoCaptionsDataset(const std::string &root_dir,
                                           int target_w,
                                           int target_h,
                                           int min_modalities,
                                           std::vector<DatasetItem>& items)
{
    items.clear();
    const fs::path root(root_dir);
    const fs::path annotations_dir = root / "annotations";
    if (!fs::exists(annotations_dir) || !fs::is_directory(annotations_dir)) return false;

    std::vector<fs::path> caption_files;
    for (const char* name : {"captions_train2017.json", "captions_val2017.json", "captions_train2014.json", "captions_val2014.json"}) {
        const fs::path p = annotations_dir / name;
        if (fs::exists(p) && fs::is_regular_file(p)) caption_files.push_back(p);
    }
    if (caption_files.empty()) return false;

    auto resolve_coco_image = [&](const std::string& file_name) -> std::string {
        const std::vector<fs::path> candidates = {
            root / file_name,
            root / "train2017" / file_name,
            root / "val2017" / file_name,
            root / "test2017" / file_name,
            root / "train2014" / file_name,
            root / "val2014" / file_name,
            root / "images" / file_name,
            root / "images" / "train2017" / file_name,
            root / "images" / "val2017" / file_name,
            root / "images" / "train2014" / file_name,
            root / "images" / "val2014" / file_name
        };
        for (const auto& c : candidates) {
            if (fs::exists(c) && fs::is_regular_file(c)) return c.string();
        }
        return std::string();
    };

    size_t total_items = 0;
    for (const auto& ann_path : caption_files) {
        try {
            std::ifstream f(ann_path);
            if (!f) continue;
            json doc;
            f >> doc;
            if (!doc.is_object() || !doc.contains("images") || !doc.contains("annotations")) continue;
            if (!doc["images"].is_array() || !doc["annotations"].is_array()) continue;

            struct CocoImageRef {
                std::string file_name;
                std::string resolved_path;
            };

            std::unordered_map<long long, CocoImageRef> images_by_id;
            images_by_id.reserve(doc["images"].size());
            for (const auto& img : doc["images"]) {
                if (!img.is_object() || !img.contains("id") || !img.contains("file_name")) continue;
                long long id = 0;
                try { id = img["id"].get<long long>(); } catch (...) { continue; }
                std::string file_name;
                try { file_name = img["file_name"].get<std::string>(); } catch (...) { continue; }
                CocoImageRef ref;
                ref.file_name = file_name;
                ref.resolved_path = resolve_coco_image(file_name);
                if (!ref.resolved_path.empty()) images_by_id.emplace(id, std::move(ref));
            }
            if (images_by_id.empty()) continue;

            std::unordered_map<long long, std::vector<std::string>> captions_by_image;
            captions_by_image.reserve(images_by_id.size());
            for (const auto& ann : doc["annotations"]) {
                if (!ann.is_object() || !ann.contains("image_id") || !ann.contains("caption")) continue;
                long long image_id = 0;
                try { image_id = ann["image_id"].get<long long>(); } catch (...) { continue; }
                auto it_img = images_by_id.find(image_id);
                if (it_img == images_by_id.end()) continue;
                std::string caption;
                try { caption = ann["caption"].get<std::string>(); } catch (...) { continue; }
                caption = sanitize_utf8(caption);
                if (!caption.empty()) captions_by_image[image_id].push_back(std::move(caption));
            }

            for (const auto& kv : captions_by_image) {
                const auto it_img = images_by_id.find(kv.first);
                if (it_img == images_by_id.end() || it_img->second.resolved_path.empty()) continue;

                std::string joined_text;
                for (size_t i = 0; i < kv.second.size(); ++i) {
                    if (i > 0) joined_text += " / ";
                    joined_text += kv.second[i];
                }
                if (joined_text.empty()) continue;

                DatasetItem item;
                item.name = fs::path(it_img->second.file_name).replace_extension("").generic_string();
                item.is_linked = true;
                item.image_file = it_img->second.resolved_path;
                item.text_inline = json{{"text", joined_text}}.dump();
                item.text_file = ann_path.string() + "#image_id=" + std::to_string(kv.first);
                item.w = target_w;
                item.h = target_h;

                if (item.countModalities() >= min_modalities) {
                    items.push_back(std::move(item));
                    total_items++;
                }
            }
        } catch (...) {
            continue;
        }
    }

    if (items.empty()) return false;

    std::cerr << "\n📂 COCO captions dataset détecté" << std::endl;
    std::cerr << "   Items indexés:      " << total_items << std::endl;
    std::cerr << "   Mode texte:         inline captions JSON" << std::endl;
    std::cerr << "   Seuil modalités:    " << min_modalities << std::endl;
    return true;
}

static inline std::vector<DatasetItem> loadDataset(const std::string &root_dir, int target_w = 64, int target_h = 64, int min_modalities = 1, const std::string& excluded_cache_path = "")
{
    static const std::regex re_image(R"(\.(png|jpg|jpeg|gif|bmp|tiff|webp|ppm|pgm)$)", std::regex::icase);
    static const std::regex re_text(R"(\.(txt|md|json|csv)$)", std::regex::icase);
    static const std::regex re_audio(R"(\.(wav|flac|mp3|ogg|m4a|aac)$)", std::regex::icase);
    static const std::regex re_video(R"(\.(mp4|mkv|avi|mov|webm|flv|ts)$)", std::regex::icase);
    
    std::vector<DatasetItem> items;
    if (root_dir.empty()) return items;

    if (loadCocoCaptionsDataset(root_dir, target_w, target_h, min_modalities, items)) {
        // Local editor annotations also override historical COCO captions.
        for (auto& item : items) {
            const fs::path sidecar = fs::path(item.image_file).replace_extension(".json");
            if (!fs::is_regular_file(sidecar)) continue;
            try {
                std::ifstream f(sidecar); json document; f >> document;
                if (document.is_object() && document.contains("captions")) {
                    (void)datasetAnnotationCaption(document, item.image_file);
                    item.annotation_file = sidecar.string();
                    item.text_file = item.annotation_file;
                    item.text_inline.clear();
                }
            } catch (const std::exception& e) {
                throw std::runtime_error("Dataset JSON " + sidecar.string() + ": " + e.what());
            }
        }
        std::cerr << "\n   ⚡ Mode: Lazy loading (images) + inline text captions" << std::endl;
        std::cerr << "   💾 RAM utilisée:    0 MB (métadonnées uniquement)" << std::endl;
        return items;
    }

    std::cerr << "\n📂 Indexation du dataset (lazy loading activé)..." << std::endl;

    // Phase 1: Indexer tous les fichiers par clé "linkable".
    // IMPORTANT: on utilise le chemin *relatif* sans extension (et pas seulement le basename)
    // pour éviter les collisions quand plusieurs sous-dossiers contiennent le même stem.
    std::unordered_map<std::string, std::vector<fs::path>> files_by_key;
    
    // `fs::relative()` canonicalise ses opérandes et suit les symlinks. Pour
    // une image liée vers un disque externe, sa clé sortait alors du dataset
    // tandis que le sidecar `.txt` gardait sa clé locale. Construire les clés
    // à partir des chemins absolus lexicaux conserve le nom du lien lui-même.
    const fs::path lexical_root = fs::absolute(fs::path(root_dir)).lexically_normal();

    auto lowerString = [](std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value;
    };
    auto stripMaskSuffix = [&](std::string stem) {
        const std::string lower = lowerString(stem);
        static const std::vector<std::string> suffixes = {
            "_mask", "-mask", ".mask", "_segmentation", "-segmentation"
        };
        for (const std::string& suffix : suffixes) {
            if (lower.size() > suffix.size() &&
                lower.compare(lower.size() - suffix.size(), suffix.size(), suffix) == 0) {
                stem.resize(stem.size() - suffix.size());
                break;
            }
        }
        return stem;
    };
    auto canonicalImageKey = [&](fs::path rel) {
        rel.replace_extension("");
        fs::path canonical;
        for (const auto& component : rel) {
            const std::string value = component.string();
            const std::string lower = lowerString(value);
            if (lower == "image" || lower == "images" || lower == "mask" || lower == "masks" ||
                lower == "segmentation" || lower == "segmentations") {
                continue;
            }
            canonical /= value;
        }
        canonical.replace_filename(stripMaskSuffix(canonical.filename().string()));
        return lowerString(canonical.generic_string());
    };
    auto isMaskPath = [&](const fs::path& rel) {
        if (stripMaskSuffix(rel.stem().string()) != rel.stem().string()) return true;
        for (const auto& component : rel.parent_path()) {
            const std::string lower = lowerString(component.string());
            if (lower == "mask" || lower == "masks" || lower == "segmentation" || lower == "segmentations") {
                return true;
            }
        }
        return false;
    };
    std::unordered_map<std::string, std::vector<fs::path>> masks_by_key;

    for (auto &p : fs::recursive_directory_iterator(root_dir))
    {
        try {
            if (!p.is_regular_file()) continue;
            const auto path = p.path();
            if (isDatasetInternalFile(path) || (!excluded_cache_path.empty() &&
                fs::absolute(path).lexically_normal() == fs::absolute(excluded_cache_path).lexically_normal())) continue;

            fs::path rel = path;
            try {
                const fs::path lexical_path = fs::absolute(path).lexically_normal();
                const fs::path candidate = lexical_path.lexically_relative(lexical_root);
                if (!candidate.empty()) rel = candidate;
            } catch (...) {
                // fallback: garder le path tel quel
            }
            if (std::regex_search(path.string(), re_image) && isMaskPath(rel)) {
                masks_by_key[canonicalImageKey(rel)].push_back(path);
                continue;
            }
            rel.replace_extension("");
            const std::string key = rel.generic_string();
            files_by_key[key].push_back(path);
        } catch (...) {
            continue;
        }
    }
    
    // Phase 2: Créer les items SANS charger les données (métadonnées uniquement)
    size_t total_linkables = 0;
    size_t valid_items = 0;
    
    auto extLower = [](const fs::path& p) -> std::string {
        std::string e = p.extension().string();
        std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return e;
    };

    auto isBetterText = [&](const fs::path& candidate, const std::string& current_path) -> bool {
        if (current_path.empty()) return true;
        const std::string ce = extLower(candidate);
        const std::string curE = extLower(fs::path(current_path));

        auto rank = [](const std::string& e) -> int {
            if (e == ".txt") return 0;
            if (e == ".md") return 1;
            if (e == ".json") return 2;
            if (e == ".csv") return 3;
            return 10;
        };
        const int r1 = rank(ce);
        const int r2 = rank(curE);
        if (r1 != r2) return r1 < r2;
        return candidate.string() < current_path;
    };

    auto isBetterImage = [&](const fs::path& candidate, const std::string& current_path) -> bool {
        if (current_path.empty()) return true;
        const std::string ce = extLower(candidate);
        const std::string curE = extLower(fs::path(current_path));

        auto rank = [](const std::string& e) -> int {
            if (e == ".png") return 0;
            if (e == ".gif") return 1;
            if (e == ".jpg" || e == ".jpeg") return 1;
            if (e == ".webp") return 2;
            if (e == ".bmp") return 3;
            if (e == ".tiff" || e == ".tif") return 4;
            if (e == ".ppm" || e == ".pgm") return 5;
            return 10;
        };
        const int r1 = rank(ce);
        const int r2 = rank(curE);
        if (r1 != r2) return r1 < r2;
        return candidate.string() < current_path;
    };

    for (const auto &[key, paths] : files_by_key)
    {
        if (paths.empty()) continue;
        
        DatasetItem item;
        item.name = key;
        item.is_linked = (paths.size() > 1);
        item.w = target_w;  // Stocker la taille cible
        item.h = target_h;
        
        std::vector<fs::path> json_candidates;
        // Index paths; JSON sidecars are inspected without retaining their data.
        for (const auto &path : paths)
        {
            const std::string pathstr = path.string();
            
            if (std::regex_search(pathstr, re_image)) {
                if (isBetterImage(path, item.image_file)) {
                    item.image_file = pathstr;
                }
            }
            else if (std::regex_search(pathstr, re_text)) {
                if (extLower(path) == ".json") { json_candidates.push_back(path); continue; }
                if (isBetterText(path, item.text_file)) {
                    item.text_file = pathstr;
                }
            }
            else if (std::regex_search(pathstr, re_audio)) {
                item.audio_file = pathstr;
            }
            else if (std::regex_search(pathstr, re_video)) {
                item.video_file = pathstr;
            }
        }

        for (const auto& path : json_candidates) {
            if (item.image_file.empty()) {
                if (isBetterText(path, item.text_file)) item.text_file = path.string();
                continue;
            }
            try {
                std::ifstream f(path);
                if (!f) throw std::runtime_error("cannot open JSON");
                json document; f >> document;
                if (document.is_object() && document.contains("captions")) {
                    (void)datasetAnnotationCaption(document, item.image_file);
                    if (item.annotation_file.empty() || path.string() < item.annotation_file) item.annotation_file = path.string();
                } else if (isBetterText(path, item.text_file)) {
                    // Generic legacy JSON remains a raw text modality.
                    item.text_file = path.string();
                }
            } catch (const std::exception& e) {
                throw std::runtime_error("Dataset JSON " + path.string() + ": " + e.what());
            }
        }
        // A verified schema takes priority over a potentially stale .txt sidecar.
        if (!item.annotation_file.empty()) item.text_file = item.annotation_file;

#ifdef MIMIR_ENABLE_FFMPEG
        if (item.audio_file.empty() && !item.video_file.empty() &&
            ffmpegMediaHasAudioStream(item.video_file)) {
            item.audio_file = item.video_file;
        }
#endif

        const auto mask_it = masks_by_key.find(canonicalImageKey(fs::path(key)));
        if (!item.image_file.empty() && mask_it != masks_by_key.end() && !mask_it->second.empty()) {
            for (const fs::path& candidate : mask_it->second) {
                if (isBetterImage(candidate, item.mask_file)) item.mask_file = candidate.string();
            }
            item.is_linked = true;
        }

        if (item.is_linked) total_linkables++;
        
        // Validation: vérifier le nombre de modalités
        int modality_count = item.countModalities();
        
        if (modality_count >= min_modalities) {
            items.push_back(std::move(item));
            valid_items++;
        }
    }
    
    // Rapport de validation (sans chargement de données)
    std::cerr << "\n📊 Dataset Indexing Report:" << std::endl;
    std::cerr << "   Items indexés:      " << valid_items << std::endl;
    std::cerr << "   Linkables détectés: " << total_linkables << std::endl;
    
    if (total_linkables > 0) {
        size_t valid_linkables = 0;
        for (const auto &item : items) {
            if (item.is_linked && (item.countModalities() >= 2 || !item.mask_file.empty())) {
                valid_linkables++;
            }
        }
        std::cerr << "   Linkables validés:  " << valid_linkables << " ✓" << std::endl;
        std::cerr << "   Ratio:              " 
                  << std::fixed << std::setprecision(1) 
                  << (100.0 * valid_linkables / total_linkables) << "%" << std::endl;
    }
    
    std::cerr << "   Seuil modalités:    " << min_modalities << std::endl;
    
    // Statistiques de complétude
    if (!items.empty()) {
        float avg_completeness = 0.0f;
        for (const auto &item : items) {
            avg_completeness += item.completeness();
        }
        avg_completeness /= items.size();
        std::cerr << "   Complétude moy.:    " 
                  << std::fixed << std::setprecision(1) 
                  << avg_completeness << "%" << std::endl;
    }
    
    std::cerr << "\n   ⚡ Mode: Lazy loading (données chargées à la demande)" << std::endl;
    std::cerr << "   💾 RAM utilisée:    0 MB (métadonnées uniquement)" << std::endl;

    return items;
}

// Cache pour datasets volumineux
struct DatasetCache {
    std::string version = "1.2";
    std::string dataset_hash; // SHA256 au lieu de uint64_t
    std::vector<DatasetItem> items;
    
    bool save(const fs::path &cache_file) const {
        try {
            json j;
            j["version"] = version;
            j["dataset_hash"] = dataset_hash;
            j["items"] = json::array();
            
            for (const auto &item : items) {
                json ji;
                ji["name"] = item.name;
                ji["is_linked"] = item.is_linked;
                ji["text_file"] = item.text_file;
                ji["text_inline"] = item.text_inline;
                ji["annotation_file"] = item.annotation_file;
                ji["image_file"] = item.image_file;
                ji["mask_file"] = item.mask_file;
                ji["audio_file"] = item.audio_file;
                ji["video_file"] = item.video_file;
                ji["w"] = item.w;
                ji["h"] = item.h;
                j["items"].push_back(ji);
            }
            
            std::ofstream f(cache_file);
            if (!f) return false;
            f << std::setw(2) << j;
            return true;
        } catch (...) {
            return false;
        }
    }
    
    static std::optional<DatasetCache> load(const fs::path &cache_file, const std::string &expected_hash) {
        try {
            if (!fs::exists(cache_file)) return std::nullopt;
            
            std::ifstream f(cache_file);
            if (!f) return std::nullopt;
            
            json j;
            f >> j;
            
            DatasetCache cache;
            cache.version = j.value("version", "1.0");
            cache.dataset_hash = j.value("dataset_hash", std::string(""));

            if (cache.version != "1.2") {
                return std::nullopt;
            }
            
            // Vérifier le hash
            if (cache.dataset_hash != expected_hash) {
                std::cerr << "⚠️  Cache invalide (hash mismatch), rechargement..." << std::endl;
                std::cerr << "   Attendu: " << expected_hash.substr(0, 16) << "..." << std::endl;
                std::cerr << "   Trouvé:  " << cache.dataset_hash.substr(0, 16) << "..." << std::endl;
                return std::nullopt;
            }
            
            if (!j.contains("items") || !j["items"].is_array()) {
                return std::nullopt;
            }
            
            for (const auto &ji : j["items"]) {
                DatasetItem item;
                item.name = ji.value("name", "");
                item.is_linked = ji.value("is_linked", false);
                item.text_file = ji.value("text_file", "");
                item.text_inline = ji.value("text_inline", "");
                item.annotation_file = ji.value("annotation_file", "");
                item.image_file = ji.value("image_file", "");
                item.mask_file = ji.value("mask_file", "");
                item.audio_file = ji.value("audio_file", "");
                item.video_file = ji.value("video_file", "");
                item.w = ji.value("w", 0);
                item.h = ji.value("h", 0);
                cache.items.push_back(std::move(item));
            }
            
            return cache;
        } catch (...) {
            return std::nullopt;
        }
    }
};

// Variante de loadDataset avec cache et gestion RAM depuis config
static inline std::vector<DatasetItem> loadDatasetCached(
    const std::string &root_dir, 
    int target_w = 64, 
    int target_h = 64, 
    int min_modalities = 1,
    const std::string &cache_path = "dataset_cache.json",
    size_t max_ram_mb = 10240,  // Paramètre RAM
    bool lazy_loading = true)    // Flag lazy loading
{
    // Initialiser le gestionnaire RAM
    auto& mgr = DatasetMemoryManager::instance();
    mgr.setMaxRAM(max_ram_mb * 1024ULL * 1024ULL);
    
    std::cerr << "\n💾 RAM Manager initialisé:" << std::endl;
    std::cerr << "   Limite:      " << max_ram_mb << " MB" << std::endl;
    std::cerr << "   Lazy loading: " << (lazy_loading ? "✓" : "✗") << std::endl;
    
    std::cerr << "\n🔐 Calcul SHA256 du dataset..." << std::endl;
    std::string dataset_hash = sha256(datasetSHA256Hash(root_dir, cache_path) + "|" +
        std::to_string(target_w) + "|" + std::to_string(target_h) + "|" + std::to_string(min_modalities));
    std::cerr << "   Hash: " << dataset_hash.substr(0, 16) << "..." << std::endl;
    
    // Essayer de charger depuis le cache
    auto cached = DatasetCache::load(cache_path, dataset_hash);
    if (cached.has_value()) {
        std::cerr << "✓ Cache valide trouvé, chargement rapide..." << std::endl;
        
        if (lazy_loading) {
            // Mode lazy: ne rien charger maintenant
            std::cerr << "  ⚡ Mode lazy: données seront chargées à la demande" << std::endl;
            std::cerr << "  💾 RAM utilisée: 0 MB (métadonnées uniquement)" << std::endl;
        } else {
            // Mode eager: charger les images immédiatement
            std::cerr << "  ⏳ Chargement des images..." << std::endl;
            size_t loaded_images = 0;
            size_t failed = 0;
            
            for (auto &item : cached->items) {
                if (!item.image_file.empty()) {
                    // Vérifier si on a assez de RAM
                    if (!mgr.canAllocate(static_cast<size_t>(target_w) * static_cast<size_t>(target_h) * 3)) {
                        std::cerr << "  ⚠️  Limite RAM atteinte, arrêt du chargement" << std::endl;
                        break;
                    }
                    
                    if (item.loadImage(target_w, target_h)) {
                        loaded_images++;
                    } else {
                        failed++;
                    }
                }
                
                // Progress tous les 100 items
                if ((loaded_images + failed) % 100 == 0) {
                    std::cerr << "  Progression: " << loaded_images << " chargées, " 
                             << failed << " échecs, RAM: " 
                             << (mgr.getCurrentRAM() / 1024 / 1024) << " MB" << std::endl;
                }
            }
            
            std::cerr << "  ✓ Images chargées: " << loaded_images << std::endl;
            if (failed > 0) {
                std::cerr << "  ⚠️  Échecs: " << failed << std::endl;
            }
            mgr.printStats();
        }
        
        return std::move(cached->items);
    }
    
    // Pas de cache ou invalide, chargement normal
    std::cerr << "⟳ Indexation du dataset..." << std::endl;
    auto items = loadDataset(root_dir, target_w, target_h, min_modalities, cache_path);
    
    // Sauvegarder le cache
    DatasetCache cache;
    cache.dataset_hash = dataset_hash;
    cache.items = items;
    
    if (cache.save(cache_path)) {
        std::cerr << "✓ Cache sauvegardé: " << cache_path << std::endl;
    } else {
        std::cerr << "⚠️  Échec sauvegarde du cache" << std::endl;
    }
    
    return items;
}

#endif // __TENSOR_HELPERS_HPP__
