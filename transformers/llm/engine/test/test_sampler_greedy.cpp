// Regression: temperature <= 0 must select the argmax for every sampler pipeline.
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <MNN/expr/ExprCreator.hpp>
#include "llm/llm.hpp"
#include "llmconfig.hpp"
#include "sampler.hpp"

using namespace MNN::Transformer;
using namespace MNN::Express;

static int runCase(const std::string& configJson, const std::vector<float>& logits, int expected) {
    auto config = std::make_shared<LlmConfig>();
    config->config_ = ujson::json::parse(configJson);
    auto context = std::make_shared<LlmContext>();
    context->status = LlmStatus::RUNNING;
    std::unique_ptr<Sampler> sampler(Sampler::createSampler(context, config));
    int failures = 0;
    for (int i = 0; i < 200; ++i) {
        auto var = _Const(logits.data(), {1, (int)logits.size()}, NCHW, halide_type_of<float>());
        int token = sampler->sample(var);
        if (token != expected) {
            ++failures;
        }
    }
    printf("%-70s -> %s (%d/200 wrong)\n", configJson.c_str(), failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}

int main() {
    const std::vector<float> logits = {1.0f, 5.0f, 2.0f, 0.5f, 4.9f};
    int bad = 0;
    bad += runCase(R"({"sampler_type":"mixed","temperature":0.0})", logits, 1);
    bad += runCase(R"({"sampler_type":"temperature","temperature":0.0})", logits, 1);
    bad += runCase(R"({"sampler_type":"mixed","mixed_samplers":["topP","temperature"],"temperature":0.0})", logits, 1);
    bad +=
        runCase(R"({"sampler_type":"mixed","mixed_samplers":["min_p","temperature"],"temperature":-1.0})", logits, 1);
    bad += runCase(R"({"sampler_type":"greedy"})", logits, 1);
    return bad == 0 ? 0 : 1;
}
