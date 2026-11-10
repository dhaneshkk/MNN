//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#include "mls_server.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <random>
#include <sstream>

namespace mls {
using MNN::Transformer::LlmStatus;

namespace {

// Model output can contain byte sequences nlohmann rejects; replace them instead of throwing.
std::string Dump(const json& j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

std::string NewCompletionId() {
    static std::atomic<uint64_t> counter{0};
    std::random_device rd;
    uint64_t r = (static_cast<uint64_t>(rd()) << 32) ^ rd();
    std::ostringstream os;
    os << "chatcmpl-" << std::hex << r << ++counter;
    return os.str();
}

json ErrorBody(const std::string& message, const std::string& type, const std::string& param = "") {
    json err = {{"message", message}, {"type", type}, {"code", nullptr}};
    err["param"] = param.empty() ? json(nullptr) : json(param);
    return json{{"error", err}};
}

void SendError(httplib::Response& res, int status, const std::string& message, const std::string& type,
               const std::string& param = "") {
    res.status = status;
    res.set_content(Dump(ErrorBody(message, type, param)), "application/json");
}

bool IsLoopback(const std::string& host) {
    return host == "127.0.0.1" || host == "localhost" || host == "::1";
}

bool ConstantTimeEquals(const std::string& a, const std::string& b) {
    unsigned char diff = a.size() == b.size() ? 0 : 1;
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; ++i) {
        diff |= static_cast<unsigned char>(a[i] ^ b[i]);
    }
    return diff == 0;
}

// Accepts plain-string content or an array of {"type":"text"} parts; other part types are rejected.
bool ContentToText(const json& content, std::string& out, std::string& error) {
    if (content.is_null()) {
        out.clear();
        return true;
    }
    if (content.is_string()) {
        out = content.get<std::string>();
        return true;
    }
    if (!content.is_array()) {
        error = "content must be a string or an array of content parts";
        return false;
    }
    out.clear();
    for (const auto& part : content) {
        if (!part.is_object() || !part.contains("type") || !part["type"].is_string()) {
            error = "each content part must be an object with a string 'type'";
            return false;
        }
        if (part["type"].get<std::string>() != "text" || !part.contains("text") || !part["text"].is_string()) {
            error = "only 'text' content parts are supported by this server";
            return false;
        }
        out += part["text"].get<std::string>();
    }
    return true;
}

std::string TrimLeadingWhitespace(const std::string& str) {
    auto it = std::find_if(str.begin(), str.end(), [](unsigned char ch) { return !std::isspace(ch); });
    return std::string(it, str.end());
}

std::string R1AssistantString(std::string content) {
    std::size_t pos = content.find("</think>");
    if (pos != std::string::npos) {
        content.erase(0, pos + std::string("</think>").length());
    }
    return TrimLeadingWhitespace(content) + "<|end_of_sentence|>";
}

std::vector<PromptItem> ConvertToR1(const std::vector<PromptItem>& chat_prompts) {
    std::vector<PromptItem> result;
    if (chat_prompts.empty()) {
        return result;
    }
    result.emplace_back("system", "<|begin_of_sentence|>You are a helpful assistant.");
    for (const auto& item : chat_prompts) {
        if (item.first == "assistant") {
            result.emplace_back("assistant", R1AssistantString(item.second));
        } else if (item.first == "user") {
            result.emplace_back("user", "<|User|>" + item.second + "<|Assistant|>");
        }
    }
    return result;
}

// Largest prefix length of s that ends on a UTF-8 code point boundary.
size_t Utf8SafeLength(const std::string& s, size_t len) {
    while (len > 0 && len < s.size() && (static_cast<unsigned char>(s[len]) & 0xC0) == 0x80) {
        --len;
    }
    return len;
}

bool ReadNumber(const json& body, const char* key, float lo, float hi, float& out, bool& present, std::string& error,
                std::string& param) {
    present = false;
    if (!body.contains(key) || body[key].is_null()) {
        return true;
    }
    if (!body[key].is_number()) {
        error = std::string(key) + " must be a number";
        param = key;
        return false;
    }
    out = body[key].get<float>();
    if (!(out >= lo && out <= hi)) {
        std::ostringstream os;
        os << key << " must be between " << lo << " and " << hi;
        error = os.str();
        param = key;
        return false;
    }
    present = true;
    return true;
}

} // namespace

bool ParseChatRequest(const json& body, ChatRequest& out, std::string& error, std::string& param) {
    if (!body.is_object()) {
        error = "request body must be a JSON object";
        return false;
    }
    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
        error = "messages must be a non-empty array";
        param = "messages";
        return false;
    }
    const auto& messages = body["messages"];
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto& m = messages[i];
        param = "messages[" + std::to_string(i) + "]";
        if (!m.is_object() || !m.contains("role") || !m["role"].is_string()) {
            error = "each message must be an object with a string 'role'";
            return false;
        }
        std::string role = m["role"].get<std::string>();
        if (role == "developer") {
            role = "system";
        }
        if (role != "system" && role != "user" && role != "assistant") {
            error = "unsupported role '" + role + "'";
            return false;
        }
        std::string text;
        if (!ContentToText(m.contains("content") ? m["content"] : json(nullptr), text, error)) {
            return false;
        }
        out.prompts.emplace_back(role, text);
    }
    param.clear();
    out.model = body.contains("model") && body["model"].is_string() ? body["model"].get<std::string>() : "";
    if (body.contains("stream") && !body["stream"].is_null()) {
        if (!body["stream"].is_boolean()) {
            error = "stream must be a boolean";
            param = "stream";
            return false;
        }
        out.stream = body["stream"].get<bool>();
    }
    if (body.contains("stream_options") && body["stream_options"].is_object()) {
        const auto& so = body["stream_options"];
        out.include_usage =
            so.contains("include_usage") && so["include_usage"].is_boolean() && so["include_usage"].get<bool>();
    }
    if (body.contains("n") && !body["n"].is_null()) {
        if (!body["n"].is_number_integer() || body["n"].get<int64_t>() != 1) {
            error = "only n=1 is supported";
            param = "n";
            return false;
        }
    }
    for (const char* key : {"max_completion_tokens", "max_tokens"}) {
        if (body.contains(key) && !body[key].is_null()) {
            if (!body[key].is_number_integer() || body[key].get<int64_t>() < 1 ||
                body[key].get<int64_t>() > (1 << 20)) {
                error = std::string(key) + " must be a positive integer";
                param = key;
                return false;
            }
            out.max_tokens = static_cast<int>(body[key].get<int64_t>());
            break;
        }
    }
    if (!ReadNumber(body, "temperature", 0.0f, 2.0f, out.temperature, out.has_temperature, error, param)) {
        return false;
    }
    if (!ReadNumber(body, "top_p", 0.0f, 1.0f, out.top_p, out.has_top_p, error, param)) {
        return false;
    }
    if (out.has_top_p && out.top_p <= 0.0f) {
        error = "top_p must be greater than 0";
        param = "top_p";
        return false;
    }
    if (body.contains("stop") && !body["stop"].is_null()) {
        const auto& stop = body["stop"];
        if (stop.is_string()) {
            out.stop.push_back(stop.get<std::string>());
        } else if (stop.is_array() && stop.size() <= 4) {
            for (const auto& s : stop) {
                if (!s.is_string()) {
                    error = "stop must be a string or an array of up to 4 strings";
                    param = "stop";
                    return false;
                }
                out.stop.push_back(s.get<std::string>());
            }
        } else {
            error = "stop must be a string or an array of up to 4 strings";
            param = "stop";
            return false;
        }
        out.stop.erase(std::remove(out.stop.begin(), out.stop.end(), std::string()), out.stop.end());
    }
    for (const char* key : {"tools", "functions"}) {
        if (body.contains(key) && !body[key].is_null() && !(body[key].is_array() && body[key].empty())) {
            error = std::string(key) + " are not supported by this server";
            param = key;
            return false;
        }
    }
    return true;
}

bool MlsServer::Authorized(const httplib::Request& req) const {
    if (options_.api_key.empty()) {
        return true;
    }
    const std::string auth = req.get_header_value("Authorization");
    const std::string prefix = "Bearer ";
    if (auth.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }
    return ConstantTimeEquals(auth.substr(prefix.size()), options_.api_key);
}

void MlsServer::AllowCors(httplib::Response& res) const {
    if (options_.cors_origin.empty()) {
        return;
    }
    res.set_header("Access-Control-Allow-Origin", options_.cors_origin);
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
    if (options_.cors_origin != "*") {
        res.set_header("Vary", "Origin");
    }
}

ChatResult MlsServer::Generate(MNN::Transformer::Llm* llm, const ChatRequest& request,
                               const std::function<bool(const std::string&)>& on_delta) {
    ChatResult result;
    std::lock_guard<std::mutex> lock(llm_mutex_);
    const auto prompts = is_r1_ ? ConvertToR1(request.prompts) : request.prompts;
    if (prompts.empty()) {
        result.error = true;
        result.error_message = "no usable messages";
        return result;
    }
    result.prompt_tokens = static_cast<int>(llm->tokenizer_encode(llm->apply_chat_template(prompts)).size());

    const bool overrideSampling = request.has_temperature || request.has_top_p;
    if (overrideSampling) {
        json cfg = {{"temperature", request.has_temperature ? request.temperature : default_temperature_},
                    {"top_p", request.has_top_p ? request.top_p : default_top_p_}};
        llm->set_config(Dump(cfg));
    }

    size_t maxStop = 0;
    for (const auto& s : request.stop) {
        maxStop = std::max(maxStop, s.size());
    }
    std::string text;
    size_t emitted = 0;
    bool stopHit = false;
    bool clientGone = false;
    auto emitUpTo = [&](size_t upTo) {
        if (upTo <= emitted) {
            return;
        }
        if (on_delta && !clientGone && !on_delta(text.substr(emitted, upTo - emitted))) {
            clientGone = true;
            llm->cancel();
        }
        emitted = upTo;
    };
    // Runs on the generating thread, so llm->cancel() here takes effect at the next decode step.
    Utf8StreamProcessor processor([&](const std::string& chunk) {
        if (stopHit || clientGone) {
            return;
        }
        const size_t searchFrom = text.size() > maxStop ? text.size() - maxStop : 0;
        text += chunk;
        size_t cut = std::string::npos;
        for (const auto& s : request.stop) {
            size_t pos = text.find(s, searchFrom);
            if (pos != std::string::npos && pos < cut) {
                cut = pos;
            }
        }
        if (cut != std::string::npos) {
            stopHit = true;
            text.resize(cut);
            emitUpTo(text.size());
            llm->cancel();
            return;
        }
        // Hold back a tail that could still grow into a stop sequence.
        const size_t hold = maxStop > 0 ? maxStop - 1 : 0;
        const size_t safe = text.size() > hold ? text.size() - hold : 0;
        emitUpTo(Utf8SafeLength(text, safe));
    });
    LlmStreamBuffer streamBuffer([&processor](const char* str, size_t len) { processor.processStream(str, len); });
    std::ostream os(&streamBuffer);

    llm->response(prompts, &os, "", request.max_tokens);
    processor.flush();
    if (!stopHit && !clientGone) {
        emitUpTo(text.size());
    }

    const auto* ctx = llm->getContext();
    const LlmStatus status = ctx->status;
    int generated = static_cast<int>(ctx->output_tokens.size());
    if (status == LlmStatus::NORMAL_FINISHED && generated > 0) {
        generated -= 1; // the stop token is recorded but not emitted
    }
    result.completion_tokens = generated;
    result.text = text;
    switch (status) {
        case LlmStatus::MAX_TOKENS_FINISHED:
        case LlmStatus::TIMEOUT:
            result.finish_reason = "length";
            break;
        case LlmStatus::INTERNAL_ERROR:
        case LlmStatus::NOT_LOADED:
            result.error = true;
            result.error_message = "generation failed (status " + std::to_string(static_cast<int>(status)) + ")";
            break;
        default:
            result.finish_reason = "stop";
            break;
    }
    // A cancelled or timed-out run leaves the engine refusing new requests until reset.
    if (status == LlmStatus::USER_CANCEL || status == LlmStatus::TIMEOUT) {
        llm->reset();
    }
    if (overrideSampling) {
        json cfg = {{"temperature", default_temperature_}, {"top_p", default_top_p_}};
        llm->set_config(Dump(cfg));
    }
    return result;
}

void MlsServer::Start(MNN::Transformer::Llm* llm, bool is_r1, const ServerOptions& options) {
    this->is_r1_ = is_r1;
    options_ = options;
    {
        json cfg = json::parse(llm->dump_config(), nullptr, false);
        if (cfg.is_object()) {
            if (cfg.contains("temperature") && cfg["temperature"].is_number()) {
                default_temperature_ = cfg["temperature"].get<float>();
            }
            if (cfg.contains("top_p") && cfg["top_p"].is_number() && cfg["top_p"].get<float>() >= 0.0f) {
                default_top_p_ = cfg["top_p"].get<float>();
            } else if (cfg.contains("topP") && cfg["topP"].is_number()) {
                default_top_p_ = cfg["topP"].get<float>();
            }
        }
    }
    if (!IsLoopback(options_.host) && options_.api_key.empty()) {
        std::cerr << "Warning: serving on " << options_.host
                  << " without an API key; anyone who can reach this port can use and reset the model.\n";
    }

    httplib::Server server;
    server.set_payload_max_length(8 * 1024 * 1024);

    server.Get("/",
               [this](const httplib::Request&, httplib::Response& res) { res.set_content(html_content, "text/html"); });

    auto models = [this](const httplib::Request& req, httplib::Response& res) {
        AllowCors(res);
        if (!Authorized(req)) {
            SendError(res, 401, "invalid or missing API key", "authentication_error");
            return;
        }
        json body = {{"object", "list"},
                     {"data", json::array({{{"id", options_.model_id},
                                            {"object", "model"},
                                            {"created", static_cast<int64_t>(std::time(nullptr))},
                                            {"owned_by", "mnn"}}})}};
        res.set_content(Dump(body), "application/json");
    };
    server.Get("/models", models);
    server.Get("/v1/models", models);

    auto reset = [this, llm](const httplib::Request& req, httplib::Response& res) {
        AllowCors(res);
        if (!Authorized(req)) {
            SendError(res, 401, "invalid or missing API key", "authentication_error");
            return;
        }
        std::lock_guard<std::mutex> lock(llm_mutex_);
        llm->reset();
        res.set_content("{\"status\": \"ok\"}", "application/json");
    };
    server.Post("/reset", reset);

    auto options_handler = [this](const httplib::Request&, httplib::Response& res) {
        AllowCors(res);
        res.status = 204;
    };
    server.Options("/chat/completions", options_handler);
    server.Options("/v1/chat/completions", options_handler);
    server.Options("/v1/models", options_handler);

    auto chat = [this, llm](const httplib::Request& req, httplib::Response& res) {
        AllowCors(res);
        if (!Authorized(req)) {
            SendError(res, 401, "invalid or missing API key", "authentication_error");
            return;
        }
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded()) {
            SendError(res, 400, "invalid JSON in request body", "invalid_request_error");
            return;
        }
        ChatRequest request;
        std::string error, param;
        if (!ParseChatRequest(body, request, error, param)) {
            SendError(res, 400, error, "invalid_request_error", param);
            return;
        }
        if (request.model.empty()) {
            request.model = options_.model_id;
        }
        if (options_.log_content) {
            std::cout << "POST " << req.path << " messages: " << Dump(body["messages"]) << std::endl;
        } else {
            std::cout << "POST " << req.path << " (" << request.prompts.size() << " messages, stream=" << request.stream
                      << ")" << std::endl;
        }
        const std::string id = NewCompletionId();
        const int64_t created = static_cast<int64_t>(std::time(nullptr));

        if (!request.stream) {
            ChatResult r = Generate(llm, request, nullptr);
            if (r.error) {
                SendError(res, 500, r.error_message, "server_error");
                return;
            }
            if (options_.log_content) {
                std::cout << "completion: " << r.text << std::endl;
            }
            json out = {{"id", id},
                        {"object", "chat.completion"},
                        {"created", created},
                        {"model", request.model},
                        {"choices", json::array({{{"index", 0},
                                                  {"message", {{"role", "assistant"}, {"content", r.text}}},
                                                  {"finish_reason", r.finish_reason}}})},
                        {"usage",
                         {{"prompt_tokens", r.prompt_tokens},
                          {"completion_tokens", r.completion_tokens},
                          {"total_tokens", r.prompt_tokens + r.completion_tokens}}}};
            res.set_content(Dump(out), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream", [this, llm, request, id,
                                                               created](size_t, httplib::DataSink& sink) {
            auto chunk = [&](const json& delta, const json& finish) {
                json c = {{"id", id},
                          {"object", "chat.completion.chunk"},
                          {"created", created},
                          {"model", request.model},
                          {"choices", json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})}};
                return "data: " + Dump(c) + "\n\n";
            };
            auto send = [&sink](const std::string& s) { return sink.is_writable() && sink.write(s.data(), s.size()); };
            if (!send(chunk({{"role", "assistant"}, {"content", ""}}, nullptr))) {
                return false;
            }
            ChatResult r = Generate(
                llm, request, [&](const std::string& delta) { return send(chunk({{"content", delta}}, nullptr)); });
            if (r.error) {
                send("data: " + Dump(ErrorBody(r.error_message, "server_error")) + "\n\n");
            } else {
                send(chunk(json::object(), r.finish_reason));
                if (request.include_usage) {
                    json u = {{"id", id},
                              {"object", "chat.completion.chunk"},
                              {"created", created},
                              {"model", request.model},
                              {"choices", json::array()},
                              {"usage",
                               {{"prompt_tokens", r.prompt_tokens},
                                {"completion_tokens", r.completion_tokens},
                                {"total_tokens", r.prompt_tokens + r.completion_tokens}}}};
                    send("data: " + Dump(u) + "\n\n");
                }
            }
            if (options_.log_content) {
                std::cout << "completion: " << r.text << std::endl;
            }
            send("data: [DONE]\n\n");
            sink.done();
            return true;
        });
    };
    server.Post("/chat/completions", chat);
    server.Post("/v1/chat/completions", chat);

    std::cout << "Starting server on http://" << options_.host << ":" << options_.port << "\n";
    if (!server.listen(options_.host, options_.port)) {
        std::cerr << "Error: Could not start server on " << options_.host << ":" << options_.port << "\n";
    }
}
} // namespace mls
