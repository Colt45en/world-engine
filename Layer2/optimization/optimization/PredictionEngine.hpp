/**
 * Prediction Engine - Pattern recognition, forecasting, and trend analysis
 * =======================================================================
 *
 * Features:
 * - Time series analysis and forecasting
 * - Pattern recognition in sequential data
 * - Anomaly detection and outlier identification
 * - Trend analysis and seasonal decomposition
 * - Machine learning-based predictions
 * - Real-time adaptive learning
 * - Multi-dimensional data analysis
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <string>
#include <vector>
#include <unordered_map>

namespace PredictionEngine {

using TimePoint = std::chrono::system_clock::time_point;
using Duration = std::chrono::nanoseconds;

struct DataPoint {
    TimePoint timestamp;
    double value;
    std::string category;
    std::unordered_map<std::string, double> features;
    std::unordered_map<std::string, std::string> metadata;
};

struct Prediction {
    double predicted_value;
    double confidence;
    Duration forecast_horizon;
    TimePoint prediction_time;
    std::string model_used;
    std::unordered_map<std::string, double> model_parameters;

    // Uncertainty bounds
    double lower_bound;
    double upper_bound;

    // Feature importance (for ML models)
    std::unordered_map<std::string, double> feature_importance;
};

struct Pattern {
    std::string pattern_id;
    std::string pattern_type; // "seasonal", "trend", "cycle", "anomaly"
    std::vector<DataPoint> pattern_data;
    double strength; // How strong/consistent the pattern is
    Duration period; // For periodic patterns
    double confidence;
    TimePoint first_detected;
    TimePoint last_seen;
    int occurrence_count;
};

struct TrendAnalysis {
    double slope;
    double intercept;
    double r_squared;
    std::string trend_direction; // "increasing", "decreasing", "stable"
    double trend_strength;
    std::vector<double> residuals;
    TimePoint analysis_start;
    TimePoint analysis_end;
};

struct AnomalyReport {
    DataPoint anomalous_point;
    double anomaly_score;
    std::string anomaly_type; // "point", "contextual", "collective"
    std::string detection_method;
    std::vector<DataPoint> context_window;
    double expected_value;
    double deviation_magnitude;
};

enum class ForecastModel {
    LINEAR_REGRESSION,
    EXPONENTIAL_SMOOTHING,
    MOVING_AVERAGE,
    ARIMA,
    SEASONAL_DECOMPOSITION,
    NEURAL_NETWORK,
    ENSEMBLE
};

enum class PatternType {
    TREND,
    SEASONAL,
    CYCLIC,
    ANOMALY,
    RECURRING_EVENT,
    SPIKE,
    DROP
};

class TimeSeriesAnalyzer {
public:
    explicit TimeSeriesAnalyzer(const std::string& series_name);

    // Data management
    void add_data_point(const DataPoint& point);
    void add_data_points(const std::vector<DataPoint>& points);
    void set_data(const std::vector<DataPoint>& data);

    // Analysis methods
    TrendAnalysis analyze_trend(Duration window = std::chrono::hours(24)) const;
    std::vector<Pattern> detect_patterns(PatternType type = PatternType::SEASONAL) const;
    std::vector<AnomalyReport> detect_anomalies(double threshold = 2.0) const;

    // Forecasting
    std::vector<Prediction> forecast(int num_points, ForecastModel model = ForecastModel::EXPONENTIAL_SMOOTHING) const;
    Prediction predict_next_value(ForecastModel model = ForecastModel::LINEAR_REGRESSION) const;

    // Statistical analysis
    double get_mean() const;
    double get_variance() const;
    double get_standard_deviation() const;
    double get_autocorrelation(int lag) const;
    std::vector<double> get_moving_average(int window_size) const;

    // Seasonal decomposition
    struct SeasonalComponents {
        std::vector<double> trend;
        std::vector<double> seasonal;
        std::vector<double> residual;
        Duration detected_period;
        double seasonal_strength;
    };

    SeasonalComponents decompose_seasonal() const;

private:
    std::string series_name_;
    std::vector<DataPoint> data_;
    mutable std::mutex data_mutex_;

    // Helper methods
    std::vector<double> extract_values() const;
    std::vector<double> smooth_data(const std::vector<double>& values, int window_size) const;
    double calculate_seasonal_strength(const std::vector<double>& seasonal) const;
};

class PatternMatcher {
public:
    PatternMatcher();

    // Pattern library management
    void add_pattern_template(const Pattern& pattern);
    void remove_pattern_template(const std::string& pattern_id);
    std::vector<Pattern> get_pattern_templates() const;

    // Pattern matching
    std::vector<Pattern> match_patterns(const std::vector<DataPoint>& data,
                                       double similarity_threshold = 0.8) const;
    Pattern find_best_match(const std::vector<DataPoint>& data) const;

    // Dynamic pattern learning
    void learn_patterns_from_data(const std::vector<DataPoint>& data,
                                 int min_pattern_length = 5,
                                 int max_pattern_length = 100);

    // Similarity calculation
    double calculate_similarity(const std::vector<DataPoint>& pattern1,
                               const std::vector<DataPoint>& pattern2) const;

private:
    std::vector<Pattern> pattern_templates_;
    mutable std::mutex patterns_mutex_;

    // Pattern extraction helpers
    std::vector<std::vector<DataPoint>> extract_subsequences(
        const std::vector<DataPoint>& data, int length) const;
    double calculate_dtw_distance(const std::vector<double>& series1,
                                 const std::vector<double>& series2) const;
};

class AnomalyDetector {
public:
    enum class Method {
        STATISTICAL,    // Z-score, IQR
        ISOLATION_FOREST,
        ONE_CLASS_SVM,
        LSTM_AUTOENCODER,
        ENSEMBLE
    };

    explicit AnomalyDetector(Method method = Method::STATISTICAL);

    // Configuration
    void set_threshold(double threshold);
    void set_window_size(int window_size);
    void set_sensitivity(double sensitivity);

    // Training (for ML-based methods)
    void train(const std::vector<DataPoint>& normal_data);
    bool is_trained() const;

    // Detection
    std::vector<AnomalyReport> detect_anomalies(const std::vector<DataPoint>& data) const;
    AnomalyReport is_anomaly(const DataPoint& point,
                            const std::vector<DataPoint>& context = {}) const;

    // Online detection
    void add_data_point(const DataPoint& point);
    std::vector<AnomalyReport> get_recent_anomalies(Duration window) const;

private:
    Method method_;
    double threshold_;
    int window_size_;
    double sensitivity_;
    bool trained_;

    std::vector<DataPoint> training_data_;
    std::vector<DataPoint> recent_data_;
    std::vector<AnomalyReport> recent_anomalies_;
    mutable std::mutex data_mutex_;

    // Method-specific implementations
    AnomalyReport detect_statistical_anomaly(const DataPoint& point,
                                            const std::vector<DataPoint>& context) const;
    double calculate_z_score(double value, const std::vector<double>& reference) const;
    std::pair<double, double> calculate_iqr_bounds(const std::vector<double>& values) const;
};

class ForecastEngine {
public:
    ForecastEngine();

    // Model management
    void register_model(ForecastModel model_type,
                       std::function<std::vector<Prediction>(const std::vector<DataPoint>&, int)> predictor);

    // Forecasting
    std::vector<Prediction> generate_forecast(const std::vector<DataPoint>& data,
                                            int num_predictions,
                                            ForecastModel model = ForecastModel::ENSEMBLE) const;

    Prediction predict_single_value(const std::vector<DataPoint>& data,
                                   ForecastModel model = ForecastModel::LINEAR_REGRESSION) const;

    // Model evaluation
    struct ModelPerformance {
        ForecastModel model;
        double mae;  // Mean Absolute Error
        double mse;  // Mean Squared Error
        double mape; // Mean Absolute Percentage Error
        double r_squared;
        std::vector<double> residuals;
    };

    ModelPerformance evaluate_model(ForecastModel model,
                                   const std::vector<DataPoint>& test_data,
                                   const std::vector<DataPoint>& training_data) const;

    // Ensemble methods
    std::vector<Prediction> ensemble_forecast(const std::vector<DataPoint>& data,
                                            int num_predictions,
                                            const std::vector<ForecastModel>& models) const;

private:
    std::unordered_map<ForecastModel, std::function<std::vector<Prediction>(const std::vector<DataPoint>&, int)>> models_;

    // Built-in model implementations
    std::vector<Prediction> linear_regression_forecast(const std::vector<DataPoint>& data, int num_predictions) const;
    std::vector<Prediction> exponential_smoothing_forecast(const std::vector<DataPoint>& data, int num_predictions) const;
    std::vector<Prediction> moving_average_forecast(const std::vector<DataPoint>& data, int num_predictions) const;

    // Utility methods
    double calculate_mae(const std::vector<double>& actual, const std::vector<double>& predicted) const;
    double calculate_mse(const std::vector<double>& actual, const std::vector<double>& predicted) const;
    double calculate_mape(const std::vector<double>& actual, const std::vector<double>& predicted) const;
};

class PredictionCore {
public:
    static PredictionCore& instance();

    // Time series management
    void create_time_series(const std::string& name);
    void remove_time_series(const std::string& name);
    TimeSeriesAnalyzer* get_time_series(const std::string& name);

    // Real-time prediction
    void add_data_point(const std::string& series_name, const DataPoint& point);
    Prediction get_latest_prediction(const std::string& series_name,
                                   ForecastModel model = ForecastModel::ENSEMBLE);

    // Pattern analysis
    std::vector<Pattern> analyze_patterns(const std::string& series_name,
                                         PatternType type = PatternType::SEASONAL);
    void learn_patterns(const std::string& series_name);

    // Anomaly monitoring
    void enable_anomaly_detection(const std::string& series_name,
                                 AnomalyDetector::Method method = AnomalyDetector::Method::STATISTICAL);
    std::vector<AnomalyReport> get_recent_anomalies(const std::string& series_name,
                                                   Duration window = std::chrono::hours(1));

    // Batch analysis
    struct AnalysisReport {
        std::string series_name;
        TrendAnalysis trend;
        std::vector<Pattern> patterns;
        std::vector<AnomalyReport> anomalies;
        std::vector<Prediction> forecast;
        TimePoint report_time;
    };

    AnalysisReport generate_analysis_report(const std::string& series_name,
                                           int forecast_points = 10);

    // Callbacks
    using PredictionCallback = std::function<void(const std::string&, const Prediction&)>;
    using AnomalyCallback = std::function<void(const std::string&, const AnomalyReport&)>;
    using PatternCallback = std::function<void(const std::string&, const Pattern&)>;

    void on_prediction_generated(PredictionCallback callback);
    void on_anomaly_detected(AnomalyCallback callback);
    void on_pattern_detected(PatternCallback callback);

private:
    PredictionCore();
    ~PredictionCore();

    std::unordered_map<std::string, std::unique_ptr<TimeSeriesAnalyzer>> time_series_;
    std::unordered_map<std::string, std::unique_ptr<AnomalyDetector>> anomaly_detectors_;
    std::unique_ptr<PatternMatcher> pattern_matcher_;
    std::unique_ptr<ForecastEngine> forecast_engine_;

    mutable std::mutex series_mutex_;

    std::vector<PredictionCallback> prediction_callbacks_;
    std::vector<AnomalyCallback> anomaly_callbacks_;
    std::vector<PatternCallback> pattern_callbacks_;
};

// Convenience macros
#define CREATE_TIME_SERIES(name) \
    PredictionEngine::PredictionCore::instance().create_time_series(name)

#define ADD_DATA_POINT(series, point) \
    PredictionEngine::PredictionCore::instance().add_data_point(series, point)

#define GET_PREDICTION(series) \
    PredictionEngine::PredictionCore::instance().get_latest_prediction(series)

} // namespace PredictionEngine
