#pragma once

#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unordered_map>
#include <unistd.h>
#include <vector>

namespace bonsai {

struct SafeTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype;
    uint64_t begin;
    uint64_t end;
    const uint8_t * data;
};

class SafetensorsFile {
public:
    SafetensorsFile() = default;
    ~SafetensorsFile() {
        if (mapped_) {
            munmap(mapped_, file_size_);
        }
    }

    SafetensorsFile(const SafetensorsFile &) = delete;
    SafetensorsFile & operator=(const SafetensorsFile &) = delete;

    void open(const std::string & path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("failed to open: " + path);
        }

        struct stat st;
        if (fstat(fd_, &st) != 0) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("failed to stat: " + path);
        }
        file_size_ = static_cast<size_t>(st.st_size);

        mapped_ = static_cast<uint8_t *>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (mapped_ == MAP_FAILED) {
            mapped_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("failed to mmap: " + path);
        }

        if (file_size_ < 8) {
            throw std::runtime_error("file too small");
        }

        uint64_t header_len = 0;
        std::memcpy(&header_len, mapped_, 8);
        if (header_len == 0 || header_len + 8 > file_size_) {
            throw std::runtime_error("invalid header length");
        }

        const char * json_begin = reinterpret_cast<const char *>(mapped_ + 8);
        std::string header(json_begin, json_begin + header_len);
        parse_header(header);

        data_start_ = 8 + header_len;
    }

    const std::unordered_map<std::string, SafeTensor> & tensors() const { return tensors_; }

    const SafeTensor * find(const std::string & name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }

    bool has(const std::string & name) const {
        return tensors_.find(name) != tensors_.end();
    }

    size_t file_size() const { return file_size_; }

    const std::string & metadata(const std::string & key, const std::string & def = "") const {
        auto it = metadata_.find(key);
        return it == metadata_.end() ? def : it->second;
    }

    static size_t dtype_bytes(const std::string & dtype) {
        if (dtype == "F32" || dtype == "I32") return 4;
        if (dtype == "F16" || dtype == "BF16") return 2;
        if (dtype == "F64" || dtype == "I64") return 8;
        if (dtype == "I8"  || dtype == "U8" || dtype == "BOOL") return 1;
        if (dtype == "I16") return 2;
        if (dtype == "I64") return 8;
        return 0;
    }

private:
    void parse_header(const std::string & json) {
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
            parse_value(key, json, pos);
            skip_ws(json, pos);
            if (json[pos] == ',') { ++pos; skip_ws(json, pos); }
        }
    }

    void parse_value(const std::string & key, const std::string & json, size_t & pos) {
        if (json[pos] != '{') throw std::runtime_error("expected '{'");
        ++pos;
        skip_ws(json, pos);

        SafeTensor t;
        t.name = key;
        bool is_metadata = (key == "__metadata__");

        while (pos < json.size() && json[pos] != '}') {
            std::string field = parse_string(json, pos);
            skip_ws(json, pos);
            if (json[pos] != ':') throw std::runtime_error("expected ':'");
            ++pos;
            skip_ws(json, pos);

            if (is_metadata) {
                parse_string(json, pos);
            } else if (field == "shape") {
                t.shape = parse_int_array(json, pos);
            } else if (field == "dtype") {
                t.dtype = parse_string(json, pos);
            } else if (field == "data_offsets") {
                auto pair = parse_int_array(json, pos);
                if (pair.size() != 2) throw std::runtime_error("data_offsets must be [a, b]");
                t.begin = static_cast<uint64_t>(pair[0]);
                t.end   = static_cast<uint64_t>(pair[1]);
            } else {
                skip_value(json, pos);
            }
            skip_ws(json, pos);
            if (json[pos] == ',') { ++pos; skip_ws(json, pos); }
        }
        if (json[pos] != '}') throw std::runtime_error("expected '}'");
        ++pos;

        if (!is_metadata) {
            t.data = mapped_ + data_start_ + t.begin;
            tensors_.emplace(key, std::move(t));
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
            if (c == '{' || c == '[' || c == '"') depth += (c == '"' ? 1 : 1), ++pos;
            else if (c == '}' || c == ']') { if (depth == 0) return; depth--; ++pos; }
            else if (c == ',' && depth == 0) return;
            else ++pos;
        } while (pos < json.size());
    }

    int fd_ = -1;
    uint8_t * mapped_ = nullptr;
    size_t file_size_ = 0;
    size_t data_start_ = 0;
    std::unordered_map<std::string, SafeTensor> tensors_;
    std::unordered_map<std::string, std::string> metadata_;
};

}  // namespace bonsai
