// Request parsing and stream helpers of the OpenAI-compatible server (no model needed).
#include <cstdio>
#include <sstream>
#include <string>
#include "mls_server.hpp"

static int gFail = 0;
#define EXPECT(cond, msg)              \
    do {                               \
        if (!(cond)) {                 \
            printf("FAIL: %s\n", msg); \
            ++gFail;                   \
        } else {                       \
            printf("PASS: %s\n", msg); \
        }                              \
    } while (0)

static bool parse(const std::string& body, mls::ChatRequest& req, std::string& err, std::string& param) {
    return mls::ParseChatRequest(json::parse(body, nullptr, false), req, err, param);
}

int main() {
    std::string err, param;
    {
        mls::ChatRequest r;
        EXPECT(parse(R"({"messages":[{"role":"user","content":"hi"}]})", r, err, param) && r.prompts.size() == 1,
               "plain string content accepted");
    }
    {
        mls::ChatRequest r;
        bool ok =
            parse(R"({"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]})",
                  r, err, param);
        EXPECT(ok && r.prompts[0].second == "ab", "text content parts concatenated");
    }
    {
        mls::ChatRequest r;
        bool ok = parse(R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]})", r,
                        err, param);
        EXPECT(!ok && param == "messages[0]", "image part rejected with 400 detail");
    }
    {
        mls::ChatRequest r;
        EXPECT(!parse(R"({"messages":[{"role":"user","content":"x"},{"content":"y"}]})", r, err, param) &&
                   param == "messages[1]",
               "malformed later message rejected, not truncated");
    }
    {
        mls::ChatRequest r;
        EXPECT(!parse(R"({"messages":[]})", r, err, param), "empty messages rejected");
    }
    {
        mls::ChatRequest r;
        EXPECT(!parse(R"({"messages":[{"role":"user","content":"x"}],"stream":"yes"})", r, err, param),
               "non-boolean stream rejected without abort");
    }
    {
        mls::ChatRequest r;
        bool ok = parse(
            R"({"messages":[{"role":"user","content":"x"}],"max_tokens":64,"temperature":0,"top_p":0.5,"stop":["\n\n","END"]})",
            r, err, param);
        EXPECT(ok && r.max_tokens == 64 && r.has_temperature && r.temperature == 0.0f && r.has_top_p &&
                   r.stop.size() == 2,
               "sampling params parsed");
    }
    {
        mls::ChatRequest r;
        EXPECT(!parse(R"({"messages":[{"role":"user","content":"x"}],"temperature":3})", r, err, param) &&
                   param == "temperature",
               "out-of-range temperature rejected");
    }
    {
        mls::ChatRequest r;
        EXPECT(!parse(R"({"messages":[{"role":"user","content":"x"}],"n":2})", r, err, param), "n>1 rejected");
    }
    {
        mls::ChatRequest r;
        EXPECT(!parse(R"({"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function"}]})", r, err, param),
               "tools rejected explicitly");
    }
    {
        std::string out;
        mls::Utf8StreamProcessor p([&out](const std::string& s) { out += s; });
        const char bytes[] = {'a', (char)0x80, 'b', (char)0xE4, (char)0xBD};
        p.processStream(bytes, sizeof(bytes));
        p.processStream("\xA0z", 2); // completes U+4F60
        EXPECT(out ==
                   "a\xEF\xBF\xBD"
                   "b\xE4\xBD\xA0z",
               "invalid byte replaced, stream keeps flowing");
    }
    {
        std::string out;
        mls::LlmStreamBuffer sb([&out](const char* s, size_t n) { out.append(s, n); });
        std::ostream os(&sb);
        os << 'x' << "yz" << std::flush;
        EXPECT(out == "xyz" && os.good(), "single-char writes do not break the stream");
    }
    printf("%s\n", gFail ? "SOME TESTS FAILED" : "ALL PASSED");
    return gFail ? 1 : 0;
}
