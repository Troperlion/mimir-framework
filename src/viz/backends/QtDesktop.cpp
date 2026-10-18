#include "../DesktopTransport.hpp"
using namespace mimir::viz::desktop;
#include "../renderers/QtScene.hpp"
#include <QApplication>
#include <QMainWindow>
#include <QToolBar>
#include <QStyleFactory>
#include <QWidget>
#include <QPainter>
#include <QImage>
#include <QTimer>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QWheelEvent>
#include <QCloseEvent>
#include <QFocusEvent>
namespace {
int keyCode(int k) {
    if(k>=Qt::Key_A && k<=Qt::Key_Z) return k;
    if(k>=Qt::Key_0 && k<=Qt::Key_9) return k;
    if(k>=Qt::Key_F1 && k<=Qt::Key_F12) return 281+k-Qt::Key_F1;
    switch(k) { case Qt::Key_Escape:return 256; case Qt::Key_Return:case Qt::Key_Enter:return 257;
      case Qt::Key_Space:return 32;case Qt::Key_Tab:return 258; case Qt::Key_Backspace:return 259;
      case Qt::Key_Left:return 260; case Qt::Key_Right:return 261; case Qt::Key_Up:return 262; case Qt::Key_Down:return 263; default:return 0; }
}
int mods(Qt::KeyboardModifiers m) { return (m.testFlag(Qt::ControlModifier)?1:0)|(m.testFlag(Qt::ShiftModifier)?2:0)|(m.testFlag(Qt::AltModifier)?4:0)|(m.testFlag(Qt::MetaModifier)?8:0); }
class Canvas final : public QWidget {
    Frame frame_; Scene scene_; QTimer timer_;
    mimir::viz::qt::ImageCache images_;
    int cursor_=-1;
    void mouse(QMouseEvent* e,int type) {
        emitEvent(Event{type,int(e->position().x()),int(e->position().y()),e->button()==Qt::RightButton?1:e->button()==Qt::MiddleButton?2:0});
    }
public:
    Canvas(const char* title) {
        (void)title; setMouseTracking(true); setFocusPolicy(Qt::StrongFocus);
        resize(800,600);
        setAttribute(Qt::WA_OpaquePaintEvent);
        connect(&timer_,&QTimer::timeout,this,[this] {
            if(ended) { QApplication::quit(); return; }
            if(takeFrame(frame_,scene_)) {
                if(size()!=QSize(int(frame_.width),int(frame_.height)))
                    setFixedSize(int(frame_.width),int(frame_.height));
                const Qt::CursorShape cursors[]={Qt::ArrowCursor,Qt::PointingHandCursor,Qt::CrossCursor,Qt::SizeFDiagCursor};
                if(cursor_!=int(frame_.cursor)) {
                    cursor_=int(frame_.cursor);
                    setCursor(cursors[frame_.cursor<4?frame_.cursor:0]);
                }
                update();
            }
        }); timer_.start(16);
    }
protected:
    bool focusNextPrevChild(bool) override { return false; }
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        if(!scene_.is_null()) mimir::viz::qt::render(p,scene_,frame_.width,frame_.height,&images_);
        else p.fillRect(rect(),QColor(32,36,43));
    }
    void mouseMoveEvent(QMouseEvent* e) override { mouse(e,Move); }
    void mousePressEvent(QMouseEvent* e) override { setFocus(Qt::MouseFocusReason); mouse(e,Down); }
    void mouseReleaseEvent(QMouseEvent* e) override { mouse(e,Up); }
    void wheelEvent(QWheelEvent* e) override { emitEvent(Event{Wheel,int(e->position().x()),int(e->position().y()),0,0,float(e->angleDelta().y())/120.f}); }
    void keyPressEvent(QKeyEvent* e) override {
        emitEvent(Event{KeyDown,0,0,keyCode(e->key()),mods(e->modifiers())});
        for(auto c:e->text().toUcs4()) emitEvent(Event{Text,0,0,int(c)});
    }
    void keyReleaseEvent(QKeyEvent* e) override { if(!e->isAutoRepeat()) emitEvent(Event{KeyUp,0,0,keyCode(e->key()),mods(e->modifiers())}); }
    void focusOutEvent(QFocusEvent*) override { emitEvent(Event{Blur}); }
    void closeEvent(QCloseEvent* e) override { emitEvent(Event{Close}); e->accept(); }
};
}
class Workbench final : public QMainWindow {
protected:
    void closeEvent(QCloseEvent* e) override { emitEvent(Event{Close});e->accept(); }
public:
    Workbench(const char* title) {
        setWindowTitle(QString::fromUtf8(title));
        auto* canvas=new Canvas(title);setCentralWidget(canvas);
        auto* toolbar=addToolBar("Workspace");toolbar->setMovable(false);toolbar->setFixedHeight(34);
        const std::pair<const char*,int> actions[]={{"Aide",'H'},{"Propriétés",'C'},{"Graphe",'N'},{"Capture",'E'},{"Actualiser",'R'}};
        for(const auto& action:actions){auto* button=toolbar->addAction(QString::fromUtf8(action.first));
            button->setToolTip(QString::fromUtf8(action.first)+" ("+QChar(action.second)+")");
            connect(button,&QAction::triggered,this,[canvas,key=action.second]{emitEvent(Event{KeyDown,0,0,key});emitEvent(Event{KeyUp,0,0,key});canvas->setFocus();});}
    }
};
int main(int argc,char** argv) {
    QApplication app(argc,argv);app.setStyle(QStyleFactory::create("Fusion"));
    app.setStyleSheet("QWidget {background:#20242b;color:#dfe5ee;font-size:12px;} QToolBar {background:#282d36;border:0;border-bottom:1px solid #101318;spacing:5px;padding:0 5px;} QToolButton {background:#313843;border:1px solid #4d5562;border-radius:5px;padding:3px 8px;color:#dfe5ee;} QToolButton:hover {background:#3e4b5c;border-color:#7e97b2;} QToolButton:pressed {background:#26394d;border-color:#6e9bc5;} QToolButton:focus {border-color:#80b4e9;} QToolTip {background:#202b38;color:#e6edf7;border:1px solid #546b84;padding:6px;}");
    Workbench workbench(argc>1?argv[1]:"Mímir Viz");workbench.show();
    emitEvent(Event{sceneProtocolHandshake}); std::thread reader(readFrames);
    int result=app.exec(); ended=true; shutdown(STDIN_FILENO,SHUT_RDWR); reader.join(); return result;
}
