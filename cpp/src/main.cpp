// Pillow-compatible resampling follows Pillow's fixed-point bilinear algorithm.
// See ../licenses/pillow.txt and ../THIRD_PARTY.md for attribution.
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <picosha2.h>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
static const std::array<std::string, 5> classes = {"bus", "bike", "car", "pedestrian", "truck"};

double milliseconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

Json read_json(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open " + path.string());
    return Json::parse(stream);
}

void write_json(const fs::path& path, const Json& value) {
    std::ofstream stream(path);
    stream << value.dump(2) << '\n';
    if (!stream) throw std::runtime_error("Cannot write " + path.string());
}

std::string file_hash(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot hash " + path.string());
    picosha2::hash256_one_by_one hasher;
    std::array<char, 65536> block{};
    while (stream) {
        stream.read(block.data(), block.size());
        hasher.process(block.begin(), block.begin() + stream.gcount());
    }
    hasher.finish();
    return picosha2::get_hash_hex_string(hasher);
}

struct Image {
    int width = 0, height = 0;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> pixels{nullptr, stbi_image_free};
    explicit Image(const fs::path& path) {
        int channels;
        pixels.reset(stbi_load(path.c_str(), &width, &height, &channels, 3));
        if (!pixels) throw std::runtime_error("Image decode failed: " + path.string() + ": " + stbi_failure_reason());
    }
};

struct Kernel {
    int stride = 0;
    std::vector<int> starts, lengths, coefficients;
    Kernel() = default;
    Kernel(int input, int output) {
        const double scale = static_cast<double>(input) / output;
        const double support = std::max(scale, 1.0);
        stride = static_cast<int>(std::ceil(support)) * 2 + 1;
        starts.resize(output);
        lengths.resize(output);
        coefficients.resize(static_cast<size_t>(output) * stride);
        std::vector<double> weights(stride);
        for (int x = 0; x < output; ++x) {
            const double center = (x + 0.5) * scale;
            const int first = std::max(0, static_cast<int>(center - support + 0.5));
            const int last = std::min(input, static_cast<int>(center + support + 0.5));
            starts[x] = first;
            lengths[x] = last - first;
            double total = 0;
            for (int k = 0; k < lengths[x]; ++k) {
                weights[k] = std::max(0.0, 1.0 - std::abs((k + first - center + 0.5) / support));
                total += weights[k];
            }
            for (int k = 0; k < lengths[x]; ++k)
                coefficients[x * stride + k] = static_cast<int>(0.5 + weights[k] / total * (1 << 22));
        }
    }
};

struct ResizePlan {
    int width, height, left, top;
    double ratio;
    Kernel horizontal, vertical;
    ResizePlan(int w, int h, int size)
        : width(0), height(0), left(0), top(0), ratio(std::min(double(size) / w, double(size) / h)) {
        // nearbyint uses ties-to-even, matching Python round for dimensions.
        width = static_cast<int>(std::nearbyint(w * ratio));
        height = static_cast<int>(std::nearbyint(h * ratio));
        if (width <= 0 || height <= 0) throw std::runtime_error("Image aspect ratio produces zero resize dimension");
        left = static_cast<int>(std::nearbyint((size - width) / 2.0 - 0.1));
        top = static_cast<int>(std::nearbyint((size - height) / 2.0 - 0.1));
        horizontal = Kernel(w, width);
        vertical = Kernel(h, height);
    }
};

struct Detection {
    float x1, y1, x2, y2, score;
    int label;
    size_t anchor;
};

float iou(const Detection& a, const Detection& b) {
    const float intersection = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
                               std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return intersection / std::max(area_a + area_b - intersection, 1e-8f);
}

class Detector {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "efficient_detection"};
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor{nullptr}, output_tensor{nullptr};
    Ort::RunOptions run_options;
    std::string input_name, output_name;
    std::vector<int64_t> input_shape, output_shape;
    std::vector<float> input, output;
    std::vector<unsigned char> horizontal_pixels, resized_pixels;
    std::map<std::pair<int, int>, ResizePlan> plans;
    std::vector<Detection> candidates, kept, class_kept;
    bool reuse;
    int size, max_detections;
    float confidence, nms_iou;
    size_t anchors;

    void preprocess(const Image& image, const ResizePlan& plan) {
        const int w = image.width, h = image.height, rw = plan.width, rh = plan.height;
        horizontal_pixels.resize(static_cast<size_t>(rw) * h * 3);
        resized_pixels.resize(static_cast<size_t>(rw) * rh * 3);
        const unsigned char* original = image.pixels.get();
        if (w == rw) {
            std::copy(original, original + static_cast<size_t>(w) * h * 3, horizontal_pixels.begin());
        } else {
            const auto& kernel = plan.horizontal;
            for (int y = 0; y < h; ++y) for (int x = 0; x < rw; ++x) {
                std::array<int, 3> sums{1 << 21, 1 << 21, 1 << 21};
                for (int k = 0; k < kernel.lengths[x]; ++k) {
                    const auto* pixel = original + (static_cast<size_t>(y) * w + kernel.starts[x] + k) * 3;
                    const int weight = kernel.coefficients[x * kernel.stride + k];
                    for (int c = 0; c < 3; ++c) sums[c] += pixel[c] * weight;
                }
                for (int c = 0; c < 3; ++c)
                    horizontal_pixels[(static_cast<size_t>(y) * rw + x) * 3 + c] = std::clamp(sums[c] >> 22, 0, 255);
            }
        }
        if (h == rh) {
            std::copy(horizontal_pixels.begin(), horizontal_pixels.end(), resized_pixels.begin());
        } else {
            const auto& kernel = plan.vertical;
            for (int y = 0; y < rh; ++y) for (int x = 0; x < rw; ++x) {
                std::array<int, 3> sums{1 << 21, 1 << 21, 1 << 21};
                for (int k = 0; k < kernel.lengths[y]; ++k) {
                    const auto* pixel = horizontal_pixels.data() + (static_cast<size_t>(kernel.starts[y] + k) * rw + x) * 3;
                    const int weight = kernel.coefficients[y * kernel.stride + k];
                    for (int c = 0; c < 3; ++c) sums[c] += pixel[c] * weight;
                }
                for (int c = 0; c < 3; ++c)
                    resized_pixels[(static_cast<size_t>(y) * rw + x) * 3 + c] = std::clamp(sums[c] >> 22, 0, 255);
            }
        }
        std::fill(input.begin(), input.end(), 114.f / 255.f);
        const size_t plane = static_cast<size_t>(size) * size;
        for (int c = 0; c < 3; ++c) for (int y = 0; y < rh; ++y) for (int x = 0; x < rw; ++x)
            input[c * plane + static_cast<size_t>(y + plan.top) * size + x + plan.left] =
                static_cast<float>(resized_pixels[(static_cast<size_t>(y) * rw + x) * 3 + c]) / 255.f;
    }

    void decode(const ResizePlan& plan, int width, int height) {
        const float* raw = output_tensor.GetTensorData<float>();
        candidates.clear();
        kept.clear();
        const float ratio = static_cast<float>(plan.ratio);
        for (size_t a = 0; a < anchors; ++a) {
            bool finite = true;
            for (size_t c = 0; c < 9; ++c) finite = finite && std::isfinite(raw[c * anchors + a]);
            if (!finite) continue;
            int label = 0;
            float score = raw[4 * anchors + a];
            for (int c = 1; c < 5; ++c) if (raw[(4 + c) * anchors + a] > score) {
                score = raw[(4 + c) * anchors + a];
                label = c;
            }
            if (score < confidence) continue;
            const float cx = raw[a], cy = raw[anchors + a], w = raw[2 * anchors + a], h = raw[3 * anchors + a];
            Detection box{
                std::clamp((cx - w / 2.f - plan.left) / ratio, 0.f, float(width)),
                std::clamp((cy - h / 2.f - plan.top) / ratio, 0.f, float(height)),
                std::clamp((cx + w / 2.f - plan.left) / ratio, 0.f, float(width)),
                std::clamp((cy + h / 2.f - plan.top) / ratio, 0.f, float(height)), score, label, a};
            if (box.x2 > box.x1 && box.y2 > box.y1) candidates.push_back(box);
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            return a.score == b.score ? a.anchor < b.anchor : a.score > b.score;
        });
        if (candidates.size() > 30000) candidates.resize(30000);
        for (int label = 0; label < 5; ++label) {
            class_kept.clear();
            for (const auto& candidate : candidates) {
                if (candidate.label != label) continue;
                bool suppressed = false;
                for (const auto& other : class_kept) if (iou(candidate, other) > nms_iou) {
                    suppressed = true;
                    break;
                }
                if (!suppressed) class_kept.push_back(candidate);
            }
            kept.insert(kept.end(), class_kept.begin(), class_kept.end());
        }
        // Python's final stable sort resolves equal scores by class, then anchor.
        std::sort(kept.begin(), kept.end(), [](const auto& a, const auto& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.label == b.label ? a.anchor < b.anchor : a.label < b.label;
        });
        if (kept.size() > static_cast<size_t>(max_detections)) kept.resize(max_detections);
    }

public:
    size_t input_allocations = 0, output_allocations = 0;
    Json timing;
    Detector(const fs::path& model, const Json& config, const Json& metadata)
        : reuse(config.value("buffer_mode", "reuse") == "reuse"), size(metadata.at("image_size")),
          max_detections(config.value("max_detections", 300)), confidence(config.value("confidence", 0.25f)),
          nms_iou(config.value("nms_iou", 0.7f)), anchors(0) {
        if (Ort::GetVersionString() != "1.31.0") throw std::runtime_error("Expected ONNX Runtime 1.31.0");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(config.value("threads", 4));
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(env, model.c_str(), options);
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
            throw std::runtime_error("Expected one input and one output");
        Ort::AllocatorWithDefaultOptions allocator;
        input_name = session.GetInputNameAllocated(0, allocator).get();
        output_name = session.GetOutputNameAllocated(0, allocator).get();
        auto in_type = session.GetInputTypeInfo(0);
        auto out_type = session.GetOutputTypeInfo(0);
        auto in = in_type.GetTensorTypeAndShapeInfo();
        auto out = out_type.GetTensorTypeAndShapeInfo();
        input_shape = in.GetShape();
        output_shape = out.GetShape();
        if (input_shape != std::vector<int64_t>{1, 3, size, size} || in.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] != 9 || output_shape[2] <= 0 ||
            out.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Expected static FP32 [1,3,S,S] -> [1,9,N] YOLO11 export");
        anchors = static_cast<size_t>(output_shape[2]);
        candidates.reserve(anchors);
        kept.reserve(anchors);
        class_kept.reserve(anchors);
        if (reuse) {
            allocate_input();
            output.resize(anchors * 9);
            output_tensor = Ort::Value::CreateTensor<float>(memory, output.data(), output.size(), output_shape.data(), output_shape.size());
            ++output_allocations;
        }
    }

    void allocate_input() {
        input_tensor = Ort::Value{nullptr};
        input = std::vector<float>(static_cast<size_t>(3) * size * size);
        input_tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), input_shape.data(), input_shape.size());
        ++input_allocations;
    }

    const std::vector<Detection>& predict(const Image& image) {
        const auto start = Clock::now();
        if (!reuse) allocate_input();
        const auto key = std::make_pair(image.width, image.height);
        if (!plans.count(key)) plans.emplace(key, ResizePlan(image.width, image.height, size));
        const auto& plan = plans.at(key);
        preprocess(image, plan);
        const auto preprocessed = Clock::now();
        const char* inputs[] = {input_name.c_str()};
        const char* outputs[] = {output_name.c_str()};
        if (reuse) {
            // Application-owned input/output Ort::Value objects and memory persist.
            session.Run(run_options, inputs, &input_tensor, 1, outputs, &output_tensor, 1);
        } else {
            auto result = session.Run(run_options, inputs, &input_tensor, 1, outputs, 1);
            output_tensor = std::move(result.front());
            ++output_allocations;
        }
        const auto inferred = Clock::now();
        decode(plan, image.width, image.height);
        const auto finished = Clock::now();
        timing = {{"preprocess", milliseconds(start, preprocessed)}, {"inference", milliseconds(preprocessed, inferred)},
                  {"postprocess", milliseconds(inferred, finished)}, {"pipeline", milliseconds(start, finished)}};
        return kept;
    }

    void dump(const fs::path& folder, size_t index) const {
        fs::create_directories(folder);
        for (const auto& entry : std::array<std::pair<std::string, std::pair<const float*, size_t>>, 2>{
                {{"input", {input.data(), input.size()}}, {"raw", {output_tensor.GetTensorData<float>(), anchors * 9}}}}) {
            std::ofstream file(folder / (std::to_string(index) + "_" + entry.first + ".f32"), std::ios::binary);
            file.write(reinterpret_cast<const char*>(entry.second.first), entry.second.second * sizeof(float));
            if (!file) throw std::runtime_error("Cannot write tensor dump");
        }
    }

    Json buffers() const {
        return {{"mode", reuse ? "reuse" : "fresh"}, {"input_bytes", input.size() * sizeof(float)},
                {"output_bytes", anchors * 9 * sizeof(float)}, {"input_tensor_allocations", input_allocations},
                {"output_tensor_allocations", output_allocations}, {"resize_plan_count", plans.size()},
                {"scope", "Application input/output tensor allocations including warm-up; excludes decoder and ORT internal allocations"}};
    }
};

Json statistics(std::vector<double> values) {
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    std::sort(values.begin(), values.end());
    auto percentile = [&](double p) {
        const double position = p * (values.size() - 1);
        const size_t index = static_cast<size_t>(position);
        return values[index] + (values[std::min(index + 1, values.size() - 1)] - values[index]) * (position - index);
    };
    return {{"mean_ms", mean}, {"p50_ms", percentile(0.5)}, {"p95_ms", percentile(0.95)}, {"p99_ms", percentile(0.99)}};
}

fs::path safe_image_path(const fs::path& root, const std::string& name) {
    const fs::path relative(name);
    if (relative.is_absolute()) throw std::runtime_error("Image names must be relative");
    for (const auto& part : relative) if (part == "..") throw std::runtime_error("Image path escapes root");
    return root / relative;
}

int main(int argc, char** argv) {
    try {
        if (argc != 3 || std::string(argv[1]) != "--config") {
            std::cerr << "Usage: efficient_infer --config configs/cpp_cpu.json\n";
            return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 2;
        }
        const auto config_path = fs::absolute(argv[2]);
        Json config = read_json(config_path);
        const auto root = fs::weakly_canonical(config_path.parent_path() / config.value("project_root", "."));
        auto path = [&](const std::string& key) { return fs::weakly_canonical(root / config.at(key).get<std::string>()); };
        const fs::path model = path("model"), images = path("images"), destination = path("output");
        if (fs::exists(destination) && !fs::is_empty(destination)) throw std::runtime_error("Choose an empty output directory");
        const auto relative_output = destination.lexically_relative(images);
        if (relative_output.empty() || *relative_output.begin() != "..") throw std::runtime_error("Output must be outside input images");
        const int threads = config.value("threads", 4), warmup = config.value("warmup", 20), repeats = config.value("repeats", 1);
        const int max_images = config.value("max_images", 0), dumps = config.value("dump_tensors", 0);
        const float confidence = config.value("confidence", 0.25f), nms = config.value("nms_iou", 0.7f);
        const auto mode = config.value("buffer_mode", "reuse");
        if (threads < 1 || warmup < 0 || repeats < 1 || max_images < 0 || dumps < 0 ||
            !std::isfinite(confidence) || confidence < 0 || confidence > 1 || !std::isfinite(nms) || nms < 0 || nms > 1 ||
            (mode != "fresh" && mode != "reuse") || config.value("max_detections", 300) < 1)
            throw std::runtime_error("Invalid thread/iteration/threshold/buffer configuration");
        const auto metadata = read_json(fs::path(model).replace_extension(".json"));
        if (metadata.at("classes") != Json(classes) || metadata.at("model_family") != "yolo11" || metadata.at("precision") != "float32" ||
            metadata.at("onnx_sha256") != file_hash(model)) throw std::runtime_error("Invalid model metadata or checksum mismatch");
        Json records = Json::array();
        if (config.contains("manifest")) records = read_json(path("manifest")).at("images");
        else if (config.contains("annotations") && !config["annotations"].is_null()) records = read_json(path("annotations")).at("images");
        else {
            std::vector<fs::path> paths;
            for (const auto& item : fs::recursive_directory_iterator(images)) {
                auto extension = item.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                if (item.is_regular_file() && (extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp"))
                    paths.push_back(item.path());
            }
            std::sort(paths.begin(), paths.end());
            for (size_t i = 0; i < paths.size(); ++i) records.push_back({{"id", i + 1}, {"file_name", fs::relative(paths[i], images).generic_string()}});
        }
        if (max_images && records.size() > static_cast<size_t>(max_images)) records.erase(records.begin() + max_images, records.end());
        if (records.empty()) throw std::runtime_error("No images");
        fs::create_directories(destination);
        const auto start = Clock::now();
        Detector detector(model, config, metadata);
        const double startup = milliseconds(start, Clock::now());
        for (int i = 0; i < warmup; ++i) {
            const auto& record = records[i % records.size()];
            Image image(safe_image_path(images, record.at("file_name")));
            detector.predict(image);
        }
        Json rows = Json::array(), predictions = Json::array();
        const auto dataset_start = Clock::now();
        for (int repeat = 0; repeat < repeats; ++repeat) for (size_t i = 0; i < records.size(); ++i) {
            const auto& record = records[i];
            const auto request_start = Clock::now();
            Image image(safe_image_path(images, record.at("file_name")));
            const auto decoded = Clock::now();
            if (record.contains("width") && (image.width != record.at("width") || image.height != record.at("height")))
                throw std::runtime_error("Image dimensions differ from manifest");
            const auto& result = detector.predict(image);
            const auto finished = Clock::now();
            Json timing = detector.timing;
            timing["image_decode"] = milliseconds(request_start, decoded);
            timing["request"] = milliseconds(request_start, finished);
            rows.push_back({{"image_id", record.at("id")}, {"file_name", record.at("file_name")},
                            {"repeat", repeat}, {"detections", result.size()}, {"timing_ms", timing}});
            if (repeat == 0) {
                for (const auto& box : result) predictions.push_back({{"image_id", record.at("id")}, {"category_id", box.label},
                    {"bbox", {double(box.x1), double(box.y1), double(box.x2) - box.x1, double(box.y2) - box.y1}}, {"score", double(box.score)}});
                if (i < static_cast<size_t>(dumps)) detector.dump(destination / "tensors", i);
            }
        }
        const double dataset_seconds = milliseconds(dataset_start, Clock::now()) / 1000;
        Json timing;
        for (const auto& stage : {"preprocess", "inference", "postprocess", "pipeline", "image_decode", "request"}) {
            std::vector<double> values;
            for (const auto& row : rows) values.push_back(row.at("timing_ms").at(stage));
            timing[stage] = statistics(values);
        }
        struct rusage resource_usage{};
        getrusage(RUSAGE_SELF, &resource_usage);
        Json summary = {{"model", model.string()}, {"onnx_sha256", metadata.at("onnx_sha256")},
            {"runtime", "cpp"}, {"onnxruntime", Ort::GetVersionString()}, {"compiler", __VERSION__},
            {"images", records.size()}, {"timed_iterations", rows.size()}, {"threads", threads},
            {"warmup", warmup}, {"batch_size", 1}, {"confidence", confidence}, {"nms_iou", nms},
            {"startup_ms", startup}, {"dataset_pass_seconds", dataset_seconds}, {"dataset_fps", rows.size() / dataset_seconds},
            {"peak_process_rss_mib", resource_usage.ru_maxrss / 1024.0}, {"buffers", detector.buffers()}, {"timing", timing},
            {"timing_scope", "Pipeline: RGB letterbox/normalization + ONNX + coordinate restoration/NMS; request adds image reading/decoding; both exclude startup, warm-up and output writing"},
            {"dataset_scope", "Image loop including prediction formatting and optional tensor dumps; excludes warm-up/startup and final JSON writing"}};
        for (const auto& stage : {"inference", "pipeline", "request"})
            summary[std::string(stage) + "_fps"] = 1000.0 / timing[stage]["mean_ms"].get<double>();
        write_json(destination / "per_image.json", rows);
        write_json(destination / "predictions.json", predictions);
        write_json(destination / "summary.json", summary);
        write_json(destination / "config.json", config);
        std::cout << "C++ " << mode << ": " << rows.size() << " predictions; pipeline FPS=" << summary["pipeline_fps"]
                  << ", request FPS=" << summary["request_fps"] << "; report: " << destination << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
