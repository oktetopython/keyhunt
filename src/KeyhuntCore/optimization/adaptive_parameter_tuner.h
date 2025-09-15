/**
 * @file adaptive_parameter_tuner.h
 * @brief Adaptive parameter tuning system for dynamic optimization
 * @author KeyhuntCUDA Team
 * 
 * T046: Advanced adaptive parameter tuning with machine learning-inspired algorithms
 */

#pragma once

#include "performance_optimizer.h"
#include <unordered_map>
#include <deque>
#include <random>

namespace keyhunt {
namespace optimization {

/**
 * @brief Parameter tuning strategy
 */
enum class TuningStrategy {
    GRADIENT_ASCENT,        // Gradient-based optimization
    GENETIC_ALGORITHM,      // Genetic algorithm approach
    SIMULATED_ANNEALING,    // Simulated annealing
    BAYESIAN_OPTIMIZATION,  // Bayesian optimization
    RANDOM_SEARCH,          // Random parameter search
    ADAPTIVE_HYBRID         // Combination of multiple strategies
};

/**
 * @brief Parameter definition for tuning
 */
struct TunableParameter {
    std::string name;
    std::string current_value;
    std::vector<std::string> possible_values;
    std::string data_type; // "int", "double", "bool", "enum"
    double min_value;
    double max_value;
    double step_size;
    double importance_weight; // 0.0 to 1.0
    std::string category; // "memory", "gpu", "threading", etc.
    
    TunableParameter()
        : min_value(0.0)
        , max_value(1.0)
        , step_size(0.1)
        , importance_weight(1.0)
    {}
};

/**
 * @brief Parameter tuning experiment result
 */
struct TuningExperiment {
    std::unordered_map<std::string, std::string> parameter_values;
    double performance_score;
    double improvement_over_baseline;
    std::chrono::milliseconds experiment_duration;
    std::chrono::system_clock::time_point timestamp;
    bool was_successful;
    std::string failure_reason;
    
    TuningExperiment()
        : performance_score(0.0)
        , improvement_over_baseline(0.0)
        , experiment_duration(std::chrono::milliseconds(0))
        , timestamp(std::chrono::system_clock::now())
        , was_successful(false)
    {}
};

/**
 * @brief Adaptive parameter tuning configuration
 */
struct AdaptiveTuningConfig {
    TuningStrategy primary_strategy;
    TuningStrategy fallback_strategy;
    size_t max_experiments_per_session;
    std::chrono::seconds experiment_timeout;
    double convergence_threshold; // Stop when improvement < threshold
    size_t population_size; // For genetic algorithm
    double mutation_rate; // For genetic algorithm
    double crossover_rate; // For genetic algorithm
    double temperature_initial; // For simulated annealing
    double temperature_decay; // For simulated annealing
    bool enable_parallel_experiments; // Run multiple experiments in parallel
    size_t max_parallel_experiments;
    
    AdaptiveTuningConfig()
        : primary_strategy(TuningStrategy::ADAPTIVE_HYBRID)
        , fallback_strategy(TuningStrategy::RANDOM_SEARCH)
        , max_experiments_per_session(50)
        , experiment_timeout(std::chrono::seconds(60))
        , convergence_threshold(0.01) // 1% improvement
        , population_size(10)
        , mutation_rate(0.1)
        , crossover_rate(0.7)
        , temperature_initial(1.0)
        , temperature_decay(0.95)
        , enable_parallel_experiments(false)
        , max_parallel_experiments(2)
    {}
};

/**
 * @brief Adaptive parameter tuner class
 */
class AdaptiveParameterTuner {
public:
    AdaptiveParameterTuner();
    ~AdaptiveParameterTuner();
    
    // Configuration and setup
    bool initialize(const AdaptiveTuningConfig& config);
    bool register_parameter(const TunableParameter& parameter);
    bool remove_parameter(const std::string& parameter_name);
    void set_performance_evaluator(std::function<double(const std::unordered_map<std::string, std::string>&)> evaluator);
    
    // Tuning operations
    bool start_adaptive_tuning();
    bool stop_adaptive_tuning();
    bool run_single_experiment(const std::unordered_map<std::string, std::string>& parameters);
    std::vector<TuningExperiment> run_parameter_sweep(const std::string& parameter_name);
    
    // Strategy-specific tuning methods
    TuningExperiment run_gradient_ascent_step();
    std::vector<TuningExperiment> run_genetic_algorithm_generation();
    TuningExperiment run_simulated_annealing_step();
    TuningExperiment run_bayesian_optimization_step();
    TuningExperiment run_random_search_step();
    
    // Results and analysis
    std::vector<TuningExperiment> get_experiment_history() const;
    TuningExperiment get_best_experiment() const;
    std::unordered_map<std::string, std::string> get_optimal_parameters() const;
    double get_current_performance_baseline() const;
    
    // Parameter management
    std::vector<TunableParameter> get_registered_parameters() const;
    bool update_parameter_bounds(const std::string& parameter_name, double min_val, double max_val);
    bool set_parameter_importance(const std::string& parameter_name, double importance);
    
    // Adaptive behavior
    bool adapt_tuning_strategy(); // Switch strategies based on performance
    bool update_parameter_priorities(); // Adjust which parameters to focus on
    bool apply_learned_constraints(); // Apply constraints learned from experiments
    
    // Advanced features
    bool enable_transfer_learning(bool enable = true); // Learn from previous sessions
    bool save_tuning_state(const std::string& filename) const;
    bool load_tuning_state(const std::string& filename);
    
    // Statistics and reporting
    struct TuningStatistics {
        size_t total_experiments;
        size_t successful_experiments;
        double best_improvement_achieved;
        double average_improvement;
        std::chrono::milliseconds total_tuning_time;
        std::unordered_map<TuningStrategy, size_t> strategy_usage_count;
        std::unordered_map<std::string, double> parameter_improvement_impact;
    };
    
    TuningStatistics get_tuning_statistics() const;
    std::string generate_tuning_report() const;

private:
    AdaptiveTuningConfig config_;
    std::vector<TunableParameter> parameters_;
    std::deque<TuningExperiment> experiment_history_;
    std::function<double(const std::unordered_map<std::string, std::string>&)> performance_evaluator_;
    
    // Tuning state
    std::atomic<bool> tuning_active_;
    std::thread tuning_thread_;
    std::atomic<bool> tuning_thread_running_;
    mutable std::mutex tuning_mutex_;
    
    // Strategy state
    TuningStrategy current_strategy_;
    std::unordered_map<std::string, std::string> baseline_parameters_;
    double baseline_performance_;
    
    // Genetic algorithm state
    std::vector<std::unordered_map<std::string, std::string>> population_;
    std::vector<double> fitness_scores_;
    
    // Simulated annealing state
    double current_temperature_;
    std::unordered_map<std::string, std::string> current_solution_;
    double current_solution_score_;
    
    // Bayesian optimization state
    std::vector<std::pair<std::unordered_map<std::string, std::string>, double>> observed_points_;
    
    // Random number generation
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;
    
    // Statistics
    TuningStatistics stats_;
    mutable std::mutex stats_mutex_;
    
    // Internal methods
    
    // Parameter handling
    std::string generate_random_parameter_value(const TunableParameter& param);
    std::string mutate_parameter_value(const TunableParameter& param, const std::string& current_value);
    bool is_valid_parameter_value(const TunableParameter& param, const std::string& value) const;
    double convert_parameter_to_double(const TunableParameter& param, const std::string& value) const;
    std::string convert_double_to_parameter(const TunableParameter& param, double value) const;
    
    // Strategy implementations
    void tuning_main_loop();
    bool execute_current_strategy();
    bool should_switch_strategy() const;
    TuningStrategy select_next_strategy() const;
    
    // Genetic algorithm helpers
    void initialize_population();
    void evaluate_population();
    std::vector<std::unordered_map<std::string, std::string>> select_parents();
    std::unordered_map<std::string, std::string> crossover(
        const std::unordered_map<std::string, std::string>& parent1,
        const std::unordered_map<std::string, std::string>& parent2
    );
    void mutate_individual(std::unordered_map<std::string, std::string>& individual);
    
    // Simulated annealing helpers
    bool accept_solution(double new_score, double old_score, double temperature) const;
    void update_temperature();
    std::unordered_map<std::string, std::string> generate_neighbor_solution(
        const std::unordered_map<std::string, std::string>& current
    );
    
    // Bayesian optimization helpers
    double predict_performance(const std::unordered_map<std::string, std::string>& parameters) const;
    std::unordered_map<std::string, std::string> select_next_candidate() const;
    double calculate_acquisition_function(const std::unordered_map<std::string, std::string>& parameters) const;
    
    // Performance evaluation
    double evaluate_parameter_set(const std::unordered_map<std::string, std::string>& parameters);
    bool is_parameter_set_valid(const std::unordered_map<std::string, std::string>& parameters) const;
    void record_experiment(const TuningExperiment& experiment);
    
    // Analysis and adaptation
    double calculate_parameter_impact(const std::string& parameter_name) const;
    std::vector<std::string> identify_most_impactful_parameters(size_t top_n = 3) const;
    bool is_converged() const;
    void update_statistics(const TuningExperiment& experiment);
    
    // Utility methods
    void cleanup_old_experiments();
    std::unordered_map<std::string, std::string> get_current_best_parameters() const;
    void log_experiment_result(const TuningExperiment& experiment) const;
};

/**
 * @brief Utility functions for adaptive tuning
 */
namespace tuning_utils {
    
    // Parameter generation utilities
    std::string generate_random_batch_size(size_t min_size = 1000000, size_t max_size = 50000000);
    std::string generate_random_thread_count(int min_threads = 64, int max_threads = 1024);
    std::string generate_random_memory_size(size_t min_mb = 100, size_t max_mb = 8192);
    
    // Performance prediction utilities
    double estimate_parameter_impact(const std::string& parameter_name, const std::string& value_change);
    bool is_parameter_combination_safe(const std::unordered_map<std::string, std::string>& parameters);
    
    // Strategy selection utilities
    TuningStrategy recommend_strategy_for_hardware(const HardwareProfile& hardware);
    TuningStrategy select_strategy_based_on_performance_history(const std::vector<TuningExperiment>& history);
    
    // Analysis utilities
    double calculate_convergence_metric(const std::vector<TuningExperiment>& recent_experiments);
    std::vector<std::string> identify_problematic_parameters(const std::vector<TuningExperiment>& experiments);
    double calculate_tuning_efficiency(const TuningStatistics& stats);
}

} // namespace optimization
} // namespace keyhunt