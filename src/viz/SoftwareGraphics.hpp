#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mimir::viz::graphics {

template<class T> struct Vector2 {
    T x{}, y{};
    Vector2() = default;
    Vector2(T x_, T y_) : x(x_), y(y_) {}
    Vector2 operator+(Vector2 b) const { return {x+b.x,y+b.y}; }
    Vector2 operator-(Vector2 b) const { return {x-b.x,y-b.y}; }
    Vector2 operator*(T factor) const { return {x*factor,y*factor}; }
    Vector2& operator+=(Vector2 b) { x+=b.x;y+=b.y;return *this; }
    bool operator==(Vector2 b) const { return x==b.x && y==b.y; }
    bool operator!=(Vector2 b) const { return !(*this==b); }
};
using Vector2f = Vector2<float>;
using Vector2i = Vector2<int>;
using Vector2u = Vector2<unsigned>;

struct FloatRect {
    Vector2f position{}, size{};
    FloatRect() = default;
    FloatRect(Vector2f p, Vector2f s) : position(p), size(s) {}
    bool contains(Vector2f point) const;
    std::optional<FloatRect> findIntersection(const FloatRect& other) const;
};

struct Color {
    uint8_t r=0,g=0,b=0,a=255;
    constexpr Color() = default;
    constexpr Color(uint8_t r_,uint8_t g_,uint8_t b_,uint8_t a_=255) : r(r_),g(g_),b(b_),a(a_) {}
    bool operator==(Color c) const { return r==c.r && g==c.g && b==c.b && a==c.a; }
    bool operator!=(Color c) const { return !(*this==c); }
    static const Color Black, White, Transparent;
};
inline const Color Color::Black{0,0,0};
inline const Color Color::White{255,255,255};
inline const Color Color::Transparent{0,0,0,0};

class Image {
    Vector2u size_{};
    std::vector<uint8_t> pixels_;
public:
    Image() = default;
    explicit Image(Vector2u size, Color color=Color::Black) { resize(size,color); }
    void resize(Vector2u size, Color color=Color::Black);
    Vector2u getSize() const { return size_; }
    const uint8_t* getPixelsPtr() const { return pixels_.data(); }
    uint8_t* pixels() { return pixels_.data(); }
    Color getPixel(Vector2u point) const;
    void setPixel(Vector2u point, Color color);
    bool loadFromFile(const std::filesystem::path& file);
    bool saveToFile(const std::filesystem::path& file) const;
    bool copy(const Image& image, Vector2u position);
};

class Texture {
    Image image_;
    bool smooth_=false;
    inline static std::atomic<uint64_t> nextRevision_{0};
    uint64_t revision_=++nextRevision_;
public:
    Texture() = default;
    explicit Texture(Vector2u size) : image_(size) {}
    bool loadFromImage(const Image& image) { image_=image;revision_=++nextRevision_;return image.getSize().x && image.getSize().y; }
    bool loadFromFile(const std::filesystem::path& file) { revision_=++nextRevision_;return image_.loadFromFile(file); }
    void setSmooth(bool smooth) { smooth_=smooth; }
    bool isSmooth() const { return smooth_; }
    Vector2u getSize() const { return image_.getSize(); }
    Image copyToImage() const { return image_; }
    const Image& image() const { return image_; }
    uint64_t revision() const { return revision_; }
    static unsigned getMaximumSize() { return 32768; }
};

class Transformable {
protected:
    Vector2f position_{},scale_{1,1},origin_{};
public:
    void setPosition(Vector2f p) { position_=p; }
    Vector2f getPosition() const { return position_; }
    void setScale(Vector2f s) { scale_=s; }
    Vector2f getScale() const { return scale_; }
    void setOrigin(Vector2f origin) { origin_=origin; }
    Vector2f getOrigin() const { return origin_; }
};

class Sprite : public Transformable {
    const Texture* texture_;
    Color color_=Color::White;
public:
    explicit Sprite(const Texture& texture) : texture_(&texture) {}
    void setTexture(const Texture& texture, bool =false) { texture_=&texture; }
    const Texture& texture() const { return *texture_; }
    void setColor(Color color) { color_=color; }
    Color color() const { return color_; }
    FloatRect getLocalBounds() const { auto s=texture_->getSize();return {{0,0},{float(s.x),float(s.y)}}; }
};

class RectangleShape : public Transformable {
    Vector2f size_;
    Color fill_=Color::White,outline_=Color::White;
    float thickness_=0;
public:
    explicit RectangleShape(Vector2f size={}) : size_(size) {}
    void setSize(Vector2f size) { size_=size; }
    Vector2f getSize() const { return size_; }
    void setFillColor(Color color) { fill_=color; }
    void setOutlineColor(Color color) { outline_=color; }
    void setOutlineThickness(float value) { thickness_=value; }
    Color fill() const { return fill_; }
    Color outline() const { return outline_; }
    float thickness() const { return thickness_; }
};

class String {
    std::u32string text_;
public:
    String() = default;
    String(const char* text);
    String(const std::string& text);
    const std::u32string& codepoints() const { return text_; }
    template<class Iterator> static String fromUtf8(Iterator begin, Iterator end) {
        return String(std::string(begin,end));
    }
};

struct FontData;
class Font {
    std::shared_ptr<FontData> data_;
    friend class Text;
    friend class RenderTexture;
public:
    bool openFromFile(const std::filesystem::path& file);
};

class Text : public Transformable {
    const Font* font_;
    String string_;
    unsigned size_=30,style_=0;
    Color fill_=Color::White;
    friend class RenderTexture;
public:
    enum { Regular=0, Bold=1 };
    explicit Text(const Font& font, const String& text={}, unsigned size=30) : font_(&font),string_(text),size_(size) {}
    void setFont(const Font& font) { font_=&font; }
    void setString(const String& text) { string_=text; }
    void setCharacterSize(unsigned size) { size_=size; }
    void setStyle(unsigned style) { style_=style; }
    void setFillColor(Color color) { fill_=color; }
    const String& string() const { return string_; }
    unsigned characterSize() const { return size_; }
    unsigned style() const { return style_; }
    Color fill() const { return fill_; }
    FloatRect getLocalBounds() const;
};

enum class PrimitiveType { LineStrip };
struct Vertex { Vector2f position; Color color=Color::White; };
class VertexArray {
    std::vector<Vertex> vertices_;
public:
    VertexArray(PrimitiveType, size_t count) : vertices_(count) {}
    Vertex& operator[](size_t i) { return vertices_[i]; }
    const std::vector<Vertex>& vertices() const { return vertices_; }
};

class View {
    FloatRect rectangle_{{0,0},{1000,1000}},viewport_{{0,0},{1,1}};
public:
    View() = default;
    explicit View(FloatRect rectangle) : rectangle_(rectangle) {}
    void setViewport(FloatRect viewport) { viewport_=viewport; }
    const FloatRect& rectangle() const { return rectangle_; }
    const FloatRect& viewport() const { return viewport_; }
};

class RenderTexture {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Texture texture_;
    View view_;
    bool recordCommands_=false;
public:
    explicit RenderTexture(Vector2u size);
    ~RenderTexture();
    RenderTexture(const RenderTexture&)=delete;
    RenderTexture& operator=(const RenderTexture&)=delete;
    bool resize(Vector2u size);
    Vector2u getSize() const;
    bool setActive(bool =true) { return true; }
    void setView(const View& view) { view_=view; }
    const View& getView() const { return view_; }
    Vector2f mapPixelToCoords(Vector2i point) const;
    void clear(Color color=Color::Black);
    void draw(const RectangleShape& shape);
    void draw(const Sprite& sprite);
    void draw(const Text& text);
    void draw(const VertexArray& vertices);
    void display();
    std::string scene() const;
    void recordCommands(bool enabled) { recordCommands_=enabled; }
    const Texture& getTexture() const { return texture_; }
};
using RenderTarget = RenderTexture;

class Time {
    float seconds_;
public:
    explicit Time(float seconds) : seconds_(seconds) {}
    float asSeconds() const { return seconds_; }
};
class Clock {
    std::chrono::steady_clock::time_point start_=std::chrono::steady_clock::now();
public:
    Time getElapsedTime() const { return Time(std::chrono::duration<float>(std::chrono::steady_clock::now()-start_).count()); }
    Time restart() { auto elapsed=getElapsedTime();start_=std::chrono::steady_clock::now();return elapsed; }
};
struct Cursor {
    enum class Type { Arrow, Hand, Cross, SizeTopLeftBottomRight };
    Type type;
    static std::optional<Cursor> createFromSystem(Type type) { return Cursor{type}; }
};
struct Keyboard {
    // Explicit values keep event logs and numeric key tests compatible with the
    // legacy adapter, without importing its headers or implementation.
    enum class Key { Unknown=-1,A=0,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,
        Num0,Num1,Num2,Num3,Num4,Num5,Num6,Num7,Num8,Num9,
        Escape=36,Space=57,Enter=58,Backspace=59,Tab=60,Left=71,Right,Up,Down,
        Numpad0=75,Numpad1,Numpad2,Numpad3,Numpad4,Numpad5,Numpad6,Numpad7,Numpad8,Numpad9,
        F1=85,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12 };
    enum class Scancode { Unknown };
};
struct Mouse {
    enum class Button { Left,Right,Middle };
    enum class Wheel { Vertical };
};
class Event {
public:
    struct Closed {};
    struct Resized { Vector2u size; };
    struct FocusLost {};
    struct TextEntered { char32_t unicode{}; };
    struct KeyPressed { Keyboard::Key code{};Keyboard::Scancode scancode{};bool alt{},control{},shift{},system{}; };
    struct KeyReleased { Keyboard::Key code{};Keyboard::Scancode scancode{};bool alt{},control{},shift{},system{}; };
    struct MouseMoved { Vector2i position; };
    struct MouseButtonPressed { Mouse::Button button{};Vector2i position; };
    struct MouseButtonReleased { Mouse::Button button{};Vector2i position; };
    struct MouseWheelScrolled { Mouse::Wheel wheel{};float delta{};Vector2i position; };
private:
    std::variant<Closed,Resized,FocusLost,TextEntered,KeyPressed,KeyReleased,MouseMoved,
        MouseButtonPressed,MouseButtonReleased,MouseWheelScrolled> value_;
public:
    template<class T> Event(T value) : value_(value) {}
    template<class T> bool is() const { return std::holds_alternative<T>(value_); }
    template<class T> const T* getIf() const { return std::get_if<T>(&value_); }
};
} // namespace mimir::viz::graphics
