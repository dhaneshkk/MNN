//
// Created by ruoyi.sjd on 2024/12/25.
// Copyright (c) 2024 Alibaba Group Holding Limited All rights reserved.
//

#pragma once
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include "llm/llm.hpp"
#include "httplib.h"
#include "jsonhpp/json.hpp"
using nlohmann::json;
using PromptItem = std::pair<std::string, std::string>;
namespace mls {

class LlmStreamBuffer : public std::streambuf {
public:
  using CallBack = std::function<void(const char* str, size_t len)>;
  explicit LlmStreamBuffer(CallBack callback) : callback_(std::move(callback)) {}
  ~LlmStreamBuffer() override = default;
protected:
  std::streamsize xsputn(const char* s, std::streamsize n) override {
    if (callback_) {
      callback_(s, n);
    }
    return n;
  }
  // Single-char writes bypass xsputn; without this the ostream goes bad after the first `<< 'c'`.
  int_type overflow(int_type ch) override {
    if (traits_type::eq_int_type(ch, traits_type::eof())) {
      return traits_type::not_eof(ch);
    }
    char c = traits_type::to_char_type(ch);
    if (callback_) {
      callback_(&c, 1);
    }
    return ch;
  }

private:
  CallBack callback_{};
};

// Re-chunks a byte stream on UTF-8 code point boundaries. Invalid bytes become U+FFFD instead of stalling.
class Utf8StreamProcessor {
  public:
    explicit Utf8StreamProcessor(std::function<void(const std::string&)> callback)
            : callback(std::move(callback)) {}

    void processStream(const char* str, size_t len) {
      utf8Buffer.append(str, len);
      std::string completeChars;
      size_t i = 0;
      while (i < utf8Buffer.size()) {
        int length = utf8CharLength(static_cast<unsigned char>(utf8Buffer[i]));
        if (length == 0 || !continuationValid(i, length)) {
          completeChars.append("\xEF\xBF\xBD");
          i += 1;
          continue;
        }
        if (i + length > utf8Buffer.size()) {
          break;
        }
        completeChars.append(utf8Buffer, i, length);
        i += length;
      }
      utf8Buffer.erase(0, i);
      if (!completeChars.empty()) {
        callback(completeChars);
      }
    }
    // Emit whatever is left (a truncated sequence) at end of generation.
    void flush() {
      if (!utf8Buffer.empty()) {
        utf8Buffer.clear();
        callback("\xEF\xBF\xBD");
      }
    }
    static int utf8CharLength(unsigned char byte) {
      if ((byte & 0x80) == 0) return 1;
      if ((byte & 0xE0) == 0xC0) return 2;
      if ((byte & 0xF0) == 0xE0) return 3;
      if ((byte & 0xF8) == 0xF0) return 4;
      return 0;
    }
  private:
    bool continuationValid(size_t start, int length) const {
      for (int k = 1; k < length && start + k < utf8Buffer.size(); ++k) {
        if ((static_cast<unsigned char>(utf8Buffer[start + k]) & 0xC0) != 0x80) {
          return false;
        }
      }
      return true;
    }
    std::string utf8Buffer;
    std::function<void(const std::string&)> callback;
  };

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 9090;
    std::string api_key;      // empty: no auth (only sensible on loopback)
    std::string cors_origin;  // empty: no CORS headers
    bool log_content = false; // log prompts and completions to stdout
    std::string model_id = "mnn";
};

// Validated /chat/completions request.
struct ChatRequest {
    std::vector<PromptItem> prompts;
    std::string model;
    bool stream = false;
    bool include_usage = false;
    int max_tokens = -1;
    bool has_temperature = false;
    float temperature = 1.0f;
    bool has_top_p = false;
    float top_p = 1.0f;
    std::vector<std::string> stop;
};

struct ChatResult {
    std::string text;
    std::string finish_reason = "stop";
    bool error = false;
    std::string error_message;
    int prompt_tokens = 0;
    int completion_tokens = 0;
};

class MlsServer {
  public:
    const char* html_content = R"""(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>MNN Frontend</title>
  <style>
    body {
      font-family: sans-serif;
      margin: 2rem;
    }
    #chat-container {
      border: 1px solid #ccc;
      border-radius: 4px;
      padding: 1rem;
      height: 400px;
      overflow-y: auto;
      margin-bottom: 1rem;
    }
    .message {
      margin: 0.5rem 0;
    }
    .user {
      color: #333;
      font-weight: bold;
    }
    .assistant {
      color: #007bff;
    }
    #user-input {
      width: 80%;
      padding: 0.5rem;
      font-size: 1rem;
    }
    .button {
      padding: 0.5rem 1rem;
      font-size: 1rem;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <h1>Chat with MNN</h1>
  <h3>MNN-LLM's server API is OpenAI API compatible. You can use other frameworks like OpenWebUI or LobeChat.</h3>
  <div id="chat-container"></div>

  <input
    type="text"
    id="user-input"
    placeholder="Type your message here..."
    onkeydown="if(event.key==='Enter'){ sendMessage(); }"
  />
  <br />
  <button id="send-btn" class="button" onclick="sendMessage()">Send</button>
  <button id="reset-btn" class="button" onclick="resetChat()">Reset</button>

  <script>
    const OPENAI_MODEL = "mnn";
    function apiKey() {
      try { return sessionStorage.getItem("mls_api_key") || ""; } catch (e) { return ""; }
    }
    function authHeaders(extra) {
      const h = Object.assign({ "Content-Type": "application/json" }, extra || {});
      const k = apiKey();
      if (k) h.Authorization = `Bearer ${k}`;
      return h;
    }
    async function authedFetch(url, init) {
      let res = await fetch(url, init);
      if (res.status === 401) {
        const k = window.prompt("API key for this server:");
        if (k) {
          try { sessionStorage.setItem("mls_api_key", k); } catch (e) {}
          init.headers = authHeaders(init.headers && init.headers.Accept ? { Accept: init.headers.Accept } : {});
          res = await fetch(url, init);
        }
      }
      return res;
    }
    let messages = [
      { role: "system", content: "You are a helpful assistant." },
    ];

    async function resetChat() {
      document.getElementById("user-input").value = "";
      document.getElementById("chat-container").innerHTML = "";
      messages = [{ role: "system", content: "You are a helpful assistant." }];

      await authedFetch("/reset", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ reset: true }),
      });
    }

    async function sendMessage() {
      const userInput = document.getElementById("user-input").value.trim();
      if (!userInput) return;

      // Display user message
      displayMessage(userInput, "user");
      document.getElementById("user-input").value = "";

      messages.push({ role: "user", content: userInput });

      try {
        // We set "stream": true to indicate we want SSE streaming from our server
        const payload = {
          model: OPENAI_MODEL,
          messages: messages,
          max_tokens: 100,
          temperature: 0.7,
          stream: true,
        };

        const response = await authedFetch("/v1/chat/completions", {
          method: "POST",
          headers: authHeaders({ Accept: "text/event-stream" }),
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          throw new Error(`Error: ${response.status} - ${response.statusText}`);
        }

        // Prepare to stream the response
        await handleStream(response);

      } catch (error) {
        displayMessage(`Error: ${error.message}`, "assistant");
      }
    }

    async function handleStream(response) {
      // We'll accumulate tokens into this variable
      let assistantMessage = "";

      // Create a DOM element for the assistant's streaming message
      const chatContainer = document.getElementById("chat-container");
      const messageElem = document.createElement("div");
      messageElem.classList.add("message", "assistant");
      const label = document.createElement("strong");
      label.className = "assistant";
      label.textContent = "Assistant: ";
      messageElem.appendChild(label);
      messageElem.appendChild(document.createElement("span"));
      chatContainer.appendChild(messageElem);
      chatContainer.scrollTop = chatContainer.scrollHeight;
      const messageTextSpan = messageElem.querySelector("span");

      // Read the response body as a stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let pending = "";

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            break;
          }

          pending += decoder.decode(value, { stream: true });
          const lines = pending.split("\n");
          pending = lines.pop();
          for (let line of lines) {
            if (!line || !line.startsWith("data: ")) {
              continue;
            }
            const jsonStr = line.substring("data: ".length).trim();
            if (jsonStr === "[DONE]") {
              messages.push({ role: "assistant", content: assistantMessage });
              return;
            }

            try {
              const parsed = JSON.parse(jsonStr);
              if (parsed.choices && parsed.choices.length > 0) {
                const delta = parsed.choices[0].delta || {};
                const deltaContent = delta.content;
                if (deltaContent) {
                  assistantMessage += deltaContent;
                  // Update the DOM text
                  messageTextSpan.textContent = assistantMessage;
                  chatContainer.scrollTop = chatContainer.scrollHeight;
                }
              }
            } catch (e) {
              console.error("Could not parse SSE line:", e, line);
            }
          }
        }
      } finally {
        reader.releaseLock();
      }
    }

    function displayMessage(text, sender) {
      const chatContainer = document.getElementById("chat-container");
      const messageElem = document.createElement("div");
      messageElem.classList.add("message", sender);

      const label = document.createElement("strong");
      label.className = sender;
      label.textContent = sender === "user" ? "User: " : "Assistant: ";
      messageElem.appendChild(label);
      messageElem.appendChild(document.createTextNode(text));

      chatContainer.appendChild(messageElem);
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  </script>
</body>
</html>
    )""";
    void Start(MNN::Transformer::Llm* llm, bool is_r1, const ServerOptions& options = ServerOptions());
    bool is_r1_{false};
private:
    // Runs one generation under llm_mutex_. on_delta (may be empty) receives text as it is produced;
    // returning false cancels generation (client went away).
    ChatResult Generate(MNN::Transformer::Llm* llm, const ChatRequest& request,
                        const std::function<bool(const std::string&)>& on_delta);
    bool Authorized(const httplib::Request& req) const;
    void AllowCors(httplib::Response& res) const;
    std::mutex llm_mutex_;
    ServerOptions options_;
    float default_temperature_ = 1.0f;
    float default_top_p_ = 0.9f;
};

// Parses an OpenAI chat request; on failure returns false and sets error/param.
bool ParseChatRequest(const json& body, ChatRequest& out, std::string& error, std::string& param);
}
