#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace bonsai {

struct SafeTensorWin {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype;
    const uint8_t * data;
};

class SafetensorsFileWin {
public:
    void open(const std::string & path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("failed to open: " + path);
        size_t size = f.tellg();
        f.seekg(0);
        file_data_.resize(size);
        f.read(reinterpret_cast<char *>(file_data_.data()), size);
        f.close();

        if (file_data_.size() < 8)
            throw std::runtime_error("file too small");

        uint64_t header_len = 0;
        std::memcpy(&header_len, file_data_.data(), 8);
        if (header_len == 0 || header_len + 8 > file_data_.size())
            throw std::runtime_error("invalid header length");

        size_t data_start_ = 8 + header_len;
        const char * json_begin = reinterpret_cast<const char *>(file_data_.data() + 8);
        std::string header(json_begin, json_begin + header_len);
        parse_header(header, data_start_);
    }

    const SafeTensorWin * find(const std::string & name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }

    const std::unordered_map<std::string, SafeTensorWin> & tensors() const {
        return tensors_;
    }

private:
    std::vector<uint8_t> file_data_;
    std::unordered_map<std::string, SafeTensorWin> tensors_;

    void parse_header(const std::string & json, size_t data_start_) {
        size_t pos = 0;
        skip_ws(json, pos);
        if (json[pos] != '{') throw std::runtime_error("header must be object");
        ++pos;
        skip_ws(json, pos);
        while (pos < json.size() && json[pos] != '}') {
            std::string key = parse_string(json, pos);
            skip_ws(json, pos);
            if (json[pos] != ':') throw std::runtime_error("expected ':'");
            ++pos;
            skip_ws(json, pos);

            if (json[pos] != '{') throw std::runtime_error("expected '{'");
            ++pos;
            skip_ws(json, pos);

            SafeTensorWin t;
            t.name = key;

            while (pos < json.size() && json[pos] != '}') {
                std::string field = parse_string(json, pos);
                skip_ws(json, pos);
                if (json[pos] != ':') throw std::runtime_error("expected ':'");
                ++pos;
                skip_ws(json, pos);

                if (field == "shape") {
                    t.shape = parse_int_array(json, pos);
                } else if (field == "dtype") {
                    t.dtype = parse_string(json, pos);
                } else if (field == "data_offsets") {
                    auto pair = parse_int_array(json, pos);
                    if (pair.size() != 2) throw std::runtime_error("data_offsets must be [a, b]");
                    t.data = file_data_.data() + data_start_ + static_cast<size_t>(pair[0]);
                } else {
                    skip_value(json, pos);
                }
                skip_ws(json, pos);
                if (json[pos] == ',') { ++pos; skip_ws(json, pos); }
            }
            if (json[pos] != '}') throw std::runtime_error("expected '}'");
            ++pos;

            tensors_.emplace(key, std::move(t));

            skip_ws(json, pos);
            if (json[pos] == ',') { ++pos; skip_ws(json, pos); }
        }
    }

    static void skip_ws(const std::string & json, size_t & pos) {
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
               json[pos] == '\n' || json[pos] == '\r')) ++pos;
    }

    static std::string parse_string(const std::string & json, size_t & pos) {
        if (json[pos] != '"') throw std::runtime_error("expected string");
        ++pos;
        std::string out;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                char c = json[pos + 1];
                if      (c == '"')  out += '"';
                else if (c == '\\') out += '\\';
                else if (c == '/')  out += '/';
                else if (c == 'n')  out += '\n';
                else if (c == 't')  out += '\t';
                else if (c == 'r')  out += '\r';
                else if (c == 'b')  out += '\b';
                else if (c == 'f')  out += '\f';
                else                out += c;
                pos += 2;
            } else {
                out += json[pos++];
            }
        }
        if (pos >= json.size()) throw std::runtime_error("unterminated string");
        ++pos;
        return out;
    }

    static std::vector<int64_t> parse_int_array(const std::string & json, size_t & pos) {
        if (json[pos] != '[') throw std::runtime_error("expected '['");
        ++pos;
        std::vector<int64_t> out;
        skip_ws(json, pos);
        while (pos < json.size() && json[pos] != ']') {
            int64_t sign = 1;
            if (json[pos] == '-') { sign = -1; ++pos; }
            int64_t v = 0;
            bool any = false;
            while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
                v = v * 10 + (json[pos] - '0');
                ++pos;
                any = true;
            }
            if (!any) throw std::runtime_error("expected number");
            out.push_back(sign * v);
            skip_ws(json, pos);
            if (json[pos] == ',') { ++pos; skip_ws(json, pos); }
        }
        if (json[pos] != ']') throw std::runtime_error("expected ']'");
        ++pos;
        return out;
    }

    static void skip_value(const std::string & json, size_t & pos) {
        int depth = 0;
        do {
            char c = json[pos];
            if (c == '{' || c == '[' || c == '"') { depth++; ++pos; }
            else if (c == '}' || c == ']') { if (depth == 0) return; depth--; ++pos; }
            else if (c == ',' && depth == 0) return;
            else ++pos;
        } while (pos < json.size());
    }
};

} // namespace bonsai
