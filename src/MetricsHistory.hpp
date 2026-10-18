#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

    struct MetricsRecord {
        int step;
        int epoch;
        int total_epochs;
        int batch;
        int total_batches;
        float loss;
        float avg_loss;
        float lr;
        int batch_time_ms;
        float bps;
        size_t memory_mb;
        double allocator_memory_mb;
        size_t params;
        float mse;
        float kl_divergence;
        float kl_beta_effective;
        float wasserstein;
        float entropy_diff;
        float moment_mismatch;
        float spatial_coherence;
        float temporal_consistency;
        float timestep;
        float grad_norm;
        float grad_max;
        int opt_type;
        int opt_step;
        float opt_beta1;
        float opt_beta2;
        float opt_eps;
        float opt_weight_decay;
        // Métriques de validation (non nulles uniquement quand is_val=true)
        float val_loss        = 0.f;
        float val_mse         = 0.f;
        int   val_step_id     = -1;  // step global auquel la validation a eu lieu
        bool  is_val          = false;
        std::string val_feedback;
    };
namespace MetricsCsv {
inline const std::string header = "step,epoch,total_epochs,batch,total_batches,loss,avg_loss,learning_rate,batch_time_ms,bps,memory_mb,allocator_memory_mb,params,mse,kl_divergence,kl_beta_effective,wasserstein,entropy_diff,moment_mismatch,spatial_coherence,temporal_consistency,timestep,grad_norm,grad_max,opt_type,opt_step,opt_beta1,opt_beta2,opt_eps,opt_weight_decay,val_loss,val_mse,val_step,val_feedback,val_rewarded,val_penalized";
inline std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> cells; std::string cell; bool quoted=false;
    for (size_t i=0; i<line.size(); ++i) {
        char c=line[i];
        if(c=='"') { if(quoted && i+1<line.size() && line[i+1]=='"') {cell+='"'; ++i;} else quoted=!quoted; }
        else if(c==',' && !quoted) {cells.push_back(cell);cell.clear();}
        else if(c!='\r') cell+=c;
    }
    cells.push_back(cell); return cells;
}
inline std::vector<MetricsRecord> read(const std::string& path) {
    std::ifstream file(path); std::string line; std::vector<MetricsRecord> records;
    if(!std::getline(file,line)) return records;
    const auto names=split(line);
    while(std::getline(file,line)) {
        const auto cells=split(line); if(cells.size()!=names.size()) continue;
        MetricsRecord r{}; bool has_loss=false;
        try { for(size_t i=0;i<names.size();++i) {
            if(cells[i].empty()) continue;
            if(names[i]=="step") {r.step=static_cast<decltype(r.step)>(std::stold(cells[i]));continue;}
            if(names[i]=="epoch") {r.epoch=static_cast<decltype(r.epoch)>(std::stold(cells[i]));continue;}
            if(names[i]=="total_epochs") {r.total_epochs=static_cast<decltype(r.total_epochs)>(std::stold(cells[i]));continue;}
            if(names[i]=="batch") {r.batch=static_cast<decltype(r.batch)>(std::stold(cells[i]));continue;}
            if(names[i]=="total_batches") {r.total_batches=static_cast<decltype(r.total_batches)>(std::stold(cells[i]));continue;}
            if(names[i]=="loss") {r.loss=static_cast<decltype(r.loss)>(std::stold(cells[i]));has_loss=true;continue;}
            if(names[i]=="avg_loss") {r.avg_loss=static_cast<decltype(r.avg_loss)>(std::stold(cells[i]));continue;}
            if(names[i]=="learning_rate") {r.lr=static_cast<decltype(r.lr)>(std::stold(cells[i]));continue;}
            if(names[i]=="batch_time_ms") {r.batch_time_ms=static_cast<decltype(r.batch_time_ms)>(std::stold(cells[i]));continue;}
            if(names[i]=="bps") {r.bps=static_cast<decltype(r.bps)>(std::stold(cells[i]));continue;}
            if(names[i]=="memory_mb") {r.memory_mb=static_cast<decltype(r.memory_mb)>(std::stold(cells[i]));continue;}
            if(names[i]=="allocator_memory_mb") {r.allocator_memory_mb=static_cast<decltype(r.allocator_memory_mb)>(std::stold(cells[i]));continue;}
            if(names[i]=="params") {r.params=static_cast<decltype(r.params)>(std::stold(cells[i]));continue;}
            if(names[i]=="mse") {r.mse=static_cast<decltype(r.mse)>(std::stold(cells[i]));continue;}
            if(names[i]=="kl_divergence") {r.kl_divergence=static_cast<decltype(r.kl_divergence)>(std::stold(cells[i]));continue;}
            if(names[i]=="kl_beta_effective") {r.kl_beta_effective=static_cast<decltype(r.kl_beta_effective)>(std::stold(cells[i]));continue;}
            if(names[i]=="wasserstein") {r.wasserstein=static_cast<decltype(r.wasserstein)>(std::stold(cells[i]));continue;}
            if(names[i]=="entropy_diff") {r.entropy_diff=static_cast<decltype(r.entropy_diff)>(std::stold(cells[i]));continue;}
            if(names[i]=="moment_mismatch") {r.moment_mismatch=static_cast<decltype(r.moment_mismatch)>(std::stold(cells[i]));continue;}
            if(names[i]=="spatial_coherence") {r.spatial_coherence=static_cast<decltype(r.spatial_coherence)>(std::stold(cells[i]));continue;}
            if(names[i]=="temporal_consistency") {r.temporal_consistency=static_cast<decltype(r.temporal_consistency)>(std::stold(cells[i]));continue;}
            if(names[i]=="timestep") {r.timestep=static_cast<decltype(r.timestep)>(std::stold(cells[i]));continue;}
            if(names[i]=="grad_norm") {r.grad_norm=static_cast<decltype(r.grad_norm)>(std::stold(cells[i]));continue;}
            if(names[i]=="grad_max") {r.grad_max=static_cast<decltype(r.grad_max)>(std::stold(cells[i]));continue;}
            if(names[i]=="opt_type") {r.opt_type=static_cast<decltype(r.opt_type)>(std::stold(cells[i]));continue;}
            if(names[i]=="opt_step") {r.opt_step=static_cast<decltype(r.opt_step)>(std::stold(cells[i]));continue;}
            if(names[i]=="opt_beta1") {r.opt_beta1=static_cast<decltype(r.opt_beta1)>(std::stold(cells[i]));continue;}
            if(names[i]=="opt_beta2") {r.opt_beta2=static_cast<decltype(r.opt_beta2)>(std::stold(cells[i]));continue;}
            if(names[i]=="opt_eps") {r.opt_eps=static_cast<decltype(r.opt_eps)>(std::stold(cells[i]));continue;}
            if(names[i]=="opt_weight_decay") {r.opt_weight_decay=static_cast<decltype(r.opt_weight_decay)>(std::stold(cells[i]));continue;}
            if(names[i]=="val_loss") {r.val_loss=static_cast<decltype(r.val_loss)>(std::stold(cells[i]));continue;}
            if(names[i]=="val_mse") {r.val_mse=static_cast<decltype(r.val_mse)>(std::stold(cells[i]));continue;}
            if(names[i]=="val_step") {r.val_step_id=static_cast<decltype(r.val_step_id)>(std::stold(cells[i]));r.is_val=true;continue;}
            if(names[i]=="val_feedback") r.val_feedback=cells[i];
        }} catch(...) {continue;}
        if(has_loss && std::isfinite(r.loss)) records.push_back(std::move(r));
    }
    return records;
}
inline void row(std::ostream& file, const MetricsRecord& r, size_t columns=36) {
    file << std::defaultfloat << std::setprecision(std::numeric_limits<float>::max_digits10);
    file << r.step << "," << r.epoch << "," << r.total_epochs << "," << r.batch << "," << r.total_batches << "," << r.loss << "," << r.avg_loss << "," << r.lr << "," << r.batch_time_ms << "," << r.bps << "," << r.memory_mb << "," << r.allocator_memory_mb << "," << r.params << "," << r.mse << "," << r.kl_divergence << "," << r.kl_beta_effective << "," << r.wasserstein << "," << r.entropy_diff << "," << r.moment_mismatch << "," << r.spatial_coherence << "," << r.temporal_consistency << "," << r.timestep << "," << r.grad_norm << "," << r.grad_max << "," << r.opt_type << "," << r.opt_step << "," << r.opt_beta1 << "," << r.opt_beta2 << "," << r.opt_eps << "," << r.opt_weight_decay;
    if(r.is_val) {
        file << ',' << r.val_loss << ',' << r.val_mse << ',' << r.val_step_id;
        if(columns>33) {
            std::string feedback; for(char c:r.val_feedback) {if(c=='"') feedback+='"'; if(c!='\n' && c!='\r') feedback+=c;}
            file << ',';
            if (feedback.find_first_of(",\"") != std::string::npos) file << '"' << feedback << '"';
            else file << feedback;
            file << ',' << (r.val_feedback=="reward") << ',' << (r.val_feedback=="penalty");
        }
    } else {for(size_t i=30;i<columns;++i) file << ',';}
    file << '\n';
}
inline bool append(const std::string& path, const MetricsRecord& r) {
    if(path.empty()) return false;
    std::error_code ec; const auto parent=std::filesystem::path(path).parent_path();
    if(!parent.empty()) std::filesystem::create_directories(parent,ec);
    std::ifstream existing(path); std::string first; std::getline(existing,first);
    const size_t columns=first.empty()?36:split(first).size();
    if(!first.empty() && first!=header && columns!=33) return false;
    bool needs_newline = false;
    if (!first.empty()) {
        existing.clear(); existing.seekg(-1, std::ios::end);
        char last = '\n'; existing.get(last); needs_newline = last != '\n';
    }
    std::ofstream file(path,std::ios::app); if(!file) return false;
    if(needs_newline) file << '\n';
    if(first.empty()) file << header << '\n';
    row(file,r,columns); file.flush(); return bool(file);
}
inline bool write(const std::string& path, const std::vector<MetricsRecord>& records) {
    std::error_code ec; const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
    std::ofstream file(path); if (!file) return false;
    file << header << '\n';
    for (const auto& record : records) row(file, record);
    return bool(file);
}
}
