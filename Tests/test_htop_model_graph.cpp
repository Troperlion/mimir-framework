#include "AsyncMonitor.hpp"
#include <iostream>
#include <stdexcept>
#ifndef _WIN32
#include <pty.h>
#include <fcntl.h>
#include <termios.h>
#endif
#define CHECK(c) do {if(!(c))throw std::runtime_error(#c);}while(0)
struct AsyncMonitorPreviewTestAccess {
#ifndef _WIN32
    static void attach(AsyncMonitor& monitor,int fd,std::shared_ptr<LiveModelConfig> config) {
        monitor.htop_=std::make_shared<HtopDisplay>(fd);
        monitor.bindRuntimeConfiguration(config);
        monitor.htop_input_fd_=::dup(fd);
        CHECK(::tcgetattr(fd,&monitor.htop_original_termios_)==0);
        auto raw=monitor.htop_original_termios_;::cfmakeraw(&raw);
        CHECK(::tcsetattr(fd,TCSANOW,&raw)==0);
        CHECK(::fcntl(monitor.htop_input_fd_,F_SETFL,O_NONBLOCK)==0);
        monitor.htop_input_ready_=true;
    }
#endif
    static void input(AsyncMonitor& monitor){monitor.processHtopInput();}
    static HtopDisplay& display(AsyncMonitor& monitor){return *monitor.htop_;}
    static bool validation(AsyncMonitor& monitor){return monitor.validation_enabled_.load();}
};
int main() {
#ifdef _WIN32
    return 77;
#else
    try {
        using Json=nlohmann::json;
        Json graph={{"mode","cost"},{"planned",true},{"training",true},{"fusion",true},{"buffer_reuse",true},
            {"nodes",Json::array()},{"edges",Json::array()},{"external_inputs",Json::array()},
            {"plan",{{"stats",{{"tensor_logical_bytes",128},{"tensor_physical_peak",64},
                {"buffer_reuse_saved",64},{"transfer_bytes",32},{"scratch_peak",16}}}}}};
        for(int i=0;i<24;++i) {
            graph["nodes"].push_back({{"index",i},{"name","layer-"+std::to_string(i)},
                {"type","Linear"},{"lane",i==1?4:0},{"depth",i==1?0:i},
                {"runtime","CUDA"},{"capability","Native"},{"parameters",12},
                {"inputs",Json::array({"x"})},{"output","y"+std::to_string(i)}});
            if(i>1)graph["edges"].push_back({{"from",i-1},{"to",i},{"tensor","y"+std::to_string(i-1)}});
        }
        auto config=std::make_shared<LiveModelConfig>();config->publishModelGraph(graph);
        Mimir::Viz::TerminalModelGraphView view;auto lines=view.lines(&graph,100,30);
        CHECK(lines.size()==30);CHECK(lines[3].find("cost")!=std::string::npos);
        CHECK(lines[4].find("scratch=16")!=std::string::npos);
        view.move(20,24);lines=view.lines(&graph,100,30);CHECK(view.pan_y>0);
        CHECK(lines[24].find("layer-20")!=std::string::npos);
        CHECK(Mimir::Viz::TerminalModelGraphView::safe("bad\033name")=="bad?name");
        CHECK(!view.lines(nullptr,100,30).empty());
        int master=-1,slave=-1;winsize size{};size.ws_col=120;size.ws_row=40;
        CHECK(::openpty(&master,&slave,nullptr,nullptr,&size)==0);
        struct Pty {int master,slave;~Pty(){::close(master);::close(slave);}} pty{master,slave};
        CHECK(::fcntl(master,F_SETFL,O_NONBLOCK)==0);
        auto drain=[&] {std::string text;char buffer[8192];int n;
            while((n=::read(master,buffer,sizeof(buffer)))>0) text.append(buffer,n);
            return text;
        };
        {
            AsyncMonitor monitor;AsyncMonitorPreviewTestAccess::attach(monitor,slave,config);
            auto send=[&](const std::string& bytes) {
                CHECK(::write(master,bytes.data(),bytes.size())==static_cast<ssize_t>(bytes.size()));
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                AsyncMonitorPreviewTestAccess::input(monitor);return drain();
            };
            auto& display=AsyncMonitorPreviewTestAccess::display(monitor);
            auto terminal=send("g");CHECK(display.modelGraphVisible());CHECK(config->modelGraphRequested());
            CHECK(terminal.find("GRAPHE DU MODELE / PLANNER")!=std::string::npos);
            CHECK(terminal.find("Planner: cost")!=std::string::npos);
            CHECK(terminal.find("Runtime: CUDA")!=std::string::npos);
            send("\033[B");CHECK(display.model_graph_view_.selected==1);
            send("\033[C");CHECK(display.model_graph_view_.pan_x>0);
            send("\033[6~");CHECK(display.model_graph_view_.selected>1);
            send("n");CHECK(!AsyncMonitorPreviewTestAccess::validation(monitor));
            send("c");CHECK(!display.configVisible());send("d");CHECK(!display.debugJsonVisible());
            send("r");CHECK(display.model_graph_view_.selected==0);
            send("g");CHECK(!display.modelGraphVisible());CHECK(!config->modelGraphRequested());
            send("n");CHECK(AsyncMonitorPreviewTestAccess::validation(monitor));
            send("c");CHECK(display.configVisible());send("g");CHECK(!display.modelGraphVisible());
            send("\033");std::this_thread::sleep_for(std::chrono::milliseconds(170));
            AsyncMonitorPreviewTestAccess::input(monitor);drain();CHECK(!display.configVisible());
            send("g");send("\033");std::this_thread::sleep_for(std::chrono::milliseconds(170));
            AsyncMonitorPreviewTestAccess::input(monitor);drain();CHECK(!display.modelGraphVisible());
            size.ws_col=24;size.ws_row=8;CHECK(::ioctl(slave,TIOCSWINSZ,&size)==0);
            terminal=send("g");CHECK(terminal.find("Graphe: agrandir")!=std::string::npos);
            send("g");
        }
        std::cout<<"HTOP_MODEL_GRAPH_PTY_OK\n";
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
#endif
}
