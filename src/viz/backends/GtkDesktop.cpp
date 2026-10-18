#include "../DesktopTransport.hpp"
using namespace mimir::viz::desktop;
#include "../renderers/GtkScene.hpp"
namespace {
Frame frame;
Scene scene;
GtkWidget* window=nullptr;
GtkWidget* canvas=nullptr;
int keyCode(guint k) {
    k=gdk_keyval_to_upper(k);
    if((k>='A'&&k<='Z')||(k>='0'&&k<='9')) return int(k);
    if(k>=GDK_KEY_KP_0 && k<=GDK_KEY_KP_9) return '0'+k-GDK_KEY_KP_0;
    if(k>=GDK_KEY_F1&&k<=GDK_KEY_F12) return 281+k-GDK_KEY_F1;
    switch(k) {case GDK_KEY_Escape:return 256;case GDK_KEY_Return:case GDK_KEY_KP_Enter:return 257;case GDK_KEY_Tab:case GDK_KEY_ISO_Left_Tab:return 258;
    case GDK_KEY_space:return 32;case GDK_KEY_BackSpace:return 259;case GDK_KEY_Left:return 260;case GDK_KEY_Right:return 261;case GDK_KEY_Up:return 262;case GDK_KEY_Down:return 263;default:return 0;}
}
gboolean draw(GtkWidget*,cairo_t* cr,gpointer) {
    if(!scene.is_null()) mimir::viz::gtk::render(cr,scene,frame.width,frame.height);
    return FALSE;
}
gboolean event(GtkWidget*,GdkEvent* e,gpointer) {
    switch(e->type) {
    case GDK_MOTION_NOTIFY:emitEvent(Event{Move,int(e->motion.x),int(e->motion.y)});return TRUE;
    case GDK_BUTTON_PRESS:case GDK_BUTTON_RELEASE:
        if(e->button.button<=3) emitEvent(Event{e->type==GDK_BUTTON_PRESS?Down:Up,int(e->button.x),int(e->button.y),e->button.button==3?1:e->button.button==2?2:0});
        return TRUE;
    case GDK_SCROLL: {
        float delta=e->scroll.direction==GDK_SCROLL_UP?1:e->scroll.direction==GDK_SCROLL_DOWN?-1:-e->scroll.delta_y;
        emitEvent(Event{Wheel,int(e->scroll.x),int(e->scroll.y),0,0,delta});return TRUE;
    }
    case GDK_KEY_PRESS:case GDK_KEY_RELEASE: {
        int m=(e->key.state&GDK_CONTROL_MASK?1:0)|(e->key.state&GDK_SHIFT_MASK?2:0)|(e->key.state&GDK_MOD1_MASK?4:0)|(e->key.state&GDK_SUPER_MASK?8:0);
        emitEvent(Event{e->type==GDK_KEY_PRESS?KeyDown:KeyUp,0,0,keyCode(e->key.keyval),m});
        if(e->type==GDK_KEY_PRESS) { auto c=gdk_keyval_to_unicode(e->key.keyval); if(e->key.keyval==GDK_KEY_BackSpace)c=8; if(c)emitEvent(Event{Text,0,0,int(c)}); } return TRUE;
    }
    case GDK_FOCUS_CHANGE:if(!e->focus_change.in)emitEvent(Event{Blur});return FALSE;
    default:return FALSE;
    }
}
gboolean tick(gpointer) {
    if(ended) { gtk_main_quit(); return G_SOURCE_REMOVE; }
    if(takeFrame(frame)) {
        scene=Scene::parse(frame.scene);
        gtk_widget_set_size_request(canvas,frame.width,frame.height); gtk_window_resize(GTK_WINDOW(window),frame.width,frame.height+34);
        const char* cursors[]={"default","pointer","crosshair","nwse-resize"};
        auto* cursor=gdk_cursor_new_from_name(gtk_widget_get_display(canvas),cursors[frame.cursor<4?frame.cursor:0]);
        gdk_window_set_cursor(gtk_widget_get_window(canvas),cursor); if(cursor)g_object_unref(cursor);
        gtk_widget_queue_draw(canvas);
    }
    return G_SOURCE_CONTINUE;
}
}
int main(int argc,char** argv) {
    if(!gtk_init_check(&argc,&argv)) return 1;
    window=gtk_window_new(GTK_WINDOW_TOPLEVEL); gtk_window_set_title(GTK_WINDOW(window),argc>1?argv[1]:"Mímir Viz");
    gtk_window_set_resizable(GTK_WINDOW(window),FALSE);
    auto* css=gtk_css_provider_new();gtk_css_provider_load_from_data(css,
        "* {font-size:12px;} window, drawingarea {background:#282828;color:#d4d4d4;} toolbar {min-height:0;background:#303030;border-bottom:1px solid #171717;padding:0 6px;} toolbutton button {min-width:0;min-height:0;margin:0;border:1px solid transparent;border-radius:3px;padding:4px 10px;background:transparent;color:#ddd;} toolbutton button:hover {background:#484848;border-color:#555;}",-1,nullptr);
    gtk_style_context_add_provider_for_screen(gdk_screen_get_default(),GTK_STYLE_PROVIDER(css),GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);g_object_unref(css);
    auto* box=gtk_box_new(GTK_ORIENTATION_VERTICAL,0);gtk_container_add(GTK_CONTAINER(window),box);
    auto* toolbar=gtk_toolbar_new();gtk_toolbar_set_style(GTK_TOOLBAR(toolbar),GTK_TOOLBAR_TEXT);gtk_widget_set_size_request(toolbar,-1,34);gtk_box_pack_start(GTK_BOX(box),toolbar,FALSE,FALSE,0);
    const std::pair<const char*,int> actions[]={{"Aide",'H'},{"Propriétés",'C'},{"Graphe",'N'},{"Capture",'E'},{"Actualiser",'R'}};
    for(const auto& action:actions){auto* button=gtk_tool_button_new(nullptr,action.first);gtk_tool_item_set_homogeneous(button,FALSE);gtk_toolbar_insert(GTK_TOOLBAR(toolbar),button,-1);
        g_signal_connect(button,"clicked",G_CALLBACK(+[](GtkToolButton*,gpointer data){const int key=GPOINTER_TO_INT(data);emitEvent(Event{KeyDown,0,0,key});emitEvent(Event{KeyUp,0,0,key});gtk_widget_grab_focus(canvas);}),GINT_TO_POINTER(action.second));}
    canvas=gtk_drawing_area_new();gtk_widget_set_can_focus(canvas,TRUE);gtk_box_pack_start(GTK_BOX(box),canvas,TRUE,TRUE,0);
    gtk_widget_add_events(canvas,GDK_POINTER_MOTION_MASK|GDK_BUTTON_PRESS_MASK|GDK_BUTTON_RELEASE_MASK|GDK_SCROLL_MASK|GDK_SMOOTH_SCROLL_MASK);
    g_signal_connect(canvas,"draw",G_CALLBACK(draw),nullptr); g_signal_connect(canvas,"event",G_CALLBACK(event),nullptr);
    g_signal_connect(window,"key-press-event",G_CALLBACK(event),nullptr); g_signal_connect(window,"key-release-event",G_CALLBACK(event),nullptr);
    g_signal_connect(window,"focus-out-event",G_CALLBACK(event),nullptr);
    g_signal_connect(window,"destroy",G_CALLBACK(+[](GtkWidget*,gpointer){emitEvent(Event{Close});gtk_main_quit();}),nullptr);
    gtk_widget_show_all(window);
    // Give the drawing area its own native event surface when supported (X11).
    gdk_window_ensure_native(gtk_widget_get_window(canvas));
    gtk_widget_grab_focus(canvas); g_timeout_add(16,tick,nullptr);
    emitEvent(Event{sceneProtocolHandshake}); std::thread reader(readFrames); gtk_main(); ended=true;
    shutdown(STDIN_FILENO,SHUT_RDWR); reader.join(); return 0;
}