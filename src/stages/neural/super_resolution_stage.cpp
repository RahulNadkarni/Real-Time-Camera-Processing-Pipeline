/**
 * @file super_resolution_stage.cpp
 * @brief Super-resolution ONNX: LR preprocess, NCHW→BGR decode, upscale, PSNR/SSIM vs downscaled HR.
 *
 * Implements `SuperResolutionStage`: ONNX load, `preprocess` downscale + 256² blob, `runInference`
 * tensor unpacking and geometric resize, `computePSNR`/`computeSSIM`, optional `process` that
 * splats cached upscale into a `Frame`, and `getCachedResult` for metrics overlay. Invoked from
 * `NeuralDispatcher` for inference; classical pipeline does not call `process` today.
 */

#include "super_resolution_stage.h"
#include "../../controls/neural_config.h"
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <memory>
#include <cmath>
#include <cstring>

struct SuperResolutionStage::Impl {
    cv::dnn::Net net;
    int run_every_n_frames{NeuralConfig::kSuperResRunEveryNFrames};
    int scale_factor{NeuralConfig::kSuperResScaleFactor};
    uint64_t frame_count{0};
    SuperResResult result;
};

SuperResolutionStage::SuperResolutionStage() : impl_(std::make_unique<Impl>()) {}

SuperResolutionStage::~SuperResolutionStage() = default;

/**
 * @brief Standard ONNX load pattern shared across neural stages.
 */
bool SuperResolutionStage::loadModel(const std::string& path) {
    try {
        impl_->net = cv::dnn::readNetFromONNX(path);
    } catch (const cv::Exception&) {
        return false;
    }
    return !impl_->net.empty();
}

/**
 * @brief LR patch construction: downscale input then `blobFromImage` to 256² (export assumes fixed spatial).
 */
cv::Mat SuperResolutionStage::preprocess(const cv::Mat& frame) {
    const int sf = impl_->scale_factor;
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(), 1.0 / sf, 1.0 / sf, cv::INTER_LINEAR);
    cv::Mat blob;
    cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, cv::Size(256, 256), cv::Scalar(0, 0, 0), true, false);
    return blob;
}

/**
 * @brief Converts 1×3×256×256 float blob to BGR 8U, resizes to `scale_factor` × original geometry.
 *
 * Computes metrics on `upscaled` resized **down** to original resolution for fair comparison.
 */
void SuperResolutionStage::runInference(const cv::Mat& frame) {
    if (frame.empty() || impl_->net.empty()) return;
    cv::Mat blob = preprocess(frame);
    impl_->net.setInput(blob);
    cv::Mat output = impl_->net.forward();
    // output blob is NCHW (1, 3, 256, 256); convert to HWC image for resize/display
    const int H = 256, W = 256, C = 3;
    if (output.total() < static_cast<size_t>(C * H * W)) return;
    cv::Mat out_img(H, W, CV_32FC3);
    const float* src = output.ptr<float>();
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float* dst = out_img.ptr<float>(y, x);
            for (int c = 0; c < C; ++c)
                dst[c] = src[c * H * W + y * W + x];
        }
    }
    out_img.convertTo(out_img, CV_8UC3, 255.0);
    cv::Mat upscaled;
    const int out_w = frame.cols * impl_->scale_factor;
    const int out_h = frame.rows * impl_->scale_factor;
    cv::resize(out_img, upscaled, cv::Size(out_w, out_h), 0, 0, cv::INTER_LINEAR);

    // PSNR/SSIM: compare at same size (resize upscaled down to original)
    cv::Mat upscaled_down;
    cv::resize(upscaled, upscaled_down, frame.size(), 0, 0, cv::INTER_LINEAR);
    float psnr = computePSNR(frame, upscaled_down);
    float ssim = computeSSIM(frame, upscaled_down);

    SuperResResult result;
    result.upscaled_frame = upscaled;
    result.psnr_db = psnr;
    result.ssim = ssim;
    result.valid = true;

    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        impl_->result = std::move(result);
        result_ready_ = true;
    }
}

/**
 * @brief MSE aggregation across channels; protects log10 domain with epsilon via early return.
 */
float SuperResolutionStage::computePSNR(const cv::Mat& original, const cv::Mat& upscaled) {
    if (original.size() != upscaled.size() || original.type() != upscaled.type()) return 0.f;
    cv::Mat o32, u32;
    original.convertTo(o32, CV_32F);
    upscaled.convertTo(u32, CV_32F);
    cv::Mat diff;
    cv::absdiff(o32, u32, diff);
    diff = diff.mul(diff);
    cv::Scalar s = cv::sum(diff);
    double mse = (s[0] + s[1] + s[2]) / (original.total() * original.channels());
    if (mse <= 1e-10) return 60.f;  // cap for near-identical
    const double max_val = 255.0;
    return static_cast<float>(10.0 * std::log10(max_val * max_val / mse));
}

/**
 * @brief Gaussian-window SSIM (Wang et al.) simplified scalar mean of SSIM map.
 */
float SuperResolutionStage::computeSSIM(const cv::Mat& original, const cv::Mat& upscaled) {
    if (original.size() != upscaled.size()) return 0.f;
    cv::Mat og, ug;
    cv::cvtColor(original, og, cv::COLOR_BGR2GRAY);
    cv::cvtColor(upscaled, ug, cv::COLOR_BGR2GRAY);
    og.convertTo(og, CV_32F);
    ug.convertTo(ug, CV_32F);
    const double C1 = 6.5025, C2 = 58.5225;  // (0.01*255)^2, (0.03*255)^2
    cv::Mat I1_2 = og.mul(og), I2_2 = ug.mul(ug), I1_I2 = og.mul(ug);
    cv::Mat mu1, mu2, mu1_2, mu2_2, mu1_mu2, sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(og, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(ug, mu2, cv::Size(11, 11), 1.5);
    mu1_2 = mu1.mul(mu1); mu2_2 = mu2.mul(mu2); mu1_mu2 = mu1.mul(mu2);
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2; sigma2_2 -= mu2_2; sigma12 -= mu1_mu2;
    cv::Mat t1 = 2 * sigma12 + C2, t2 = sigma1_2 + sigma2_2 + C2;
    cv::Mat ssim_map;
    cv::divide(t1, t2, ssim_map);
    cv::Scalar m = cv::mean(ssim_map);
    return static_cast<float>(m[0]);
}

/**
 * @brief Optional path: if cache holds upscaled Mat, memcpy into `frame` and update geometry.
 *
 * Not called from `Pipeline::run_stage` in this repo — documented for completeness if rewired.
 */
void SuperResolutionStage::process(Frame& frame, int64_t* out_latency_us) {
    if (out_latency_us) *out_latency_us = 0;
    // Inference is run by NeuralDispatcher on a background thread; we only consume cached result.
    impl_->frame_count++;
    std::lock_guard<std::mutex> lock(result_mutex_);
    if (result_ready_ && !impl_->result.upscaled_frame.empty()) {
        const cv::Mat& up = impl_->result.upscaled_frame;
        const size_t n = up.total() * up.elemSize();
        frame.buffer.resize(n);
        std::memcpy(frame.buffer.data(), up.data, n);
        frame.width = up.cols;
        frame.height = up.rows;
        frame.channels = up.channels();
    }
}

/**
 * @brief Mutex copy of last `SuperResResult` (Mat uses refcounted header; shared pixel storage until reassigned).
 *
 * Display uses metrics text; full image is available if you extend the renderer to show SR output.
 */
SuperResResult SuperResolutionStage::getCachedResult() {
    std::lock_guard<std::mutex> lock(result_mutex_);
    if (!result_ready_) return SuperResResult();
    return impl_->result;
}

const char* SuperResolutionStage::name() const {
    return "SuperResolution";
}
