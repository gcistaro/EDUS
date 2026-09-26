#ifndef FORTRANBINARY_HPP
#define FORTRANBINARY_HPP

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

/// @brief Minimal reader/writer for Fortran sequential unformatted files
/// (gfortran / ifort convention: every record is surrounded by 4-byte markers
/// with its length in bytes). Records longer than 2 GiB are split by gfortran
/// in sub-records, signalled by a negative marker: they are handled as well.
class FortranBinaryReader
{
    private:
        std::ifstream file_;
        std::string filename_;
        std::vector<char> record_;
        size_t position_ = 0;

    public:
        explicit FortranBinaryReader(const std::string& filename__)
            : file_(filename__, std::ios::binary), filename_(filename__)
        {
            if (!file_.is_open()) {
                throw std::runtime_error("FortranBinaryReader: cannot open file " + filename__);
            }
        }

        /// Read the next record in memory. After this call the typed getters
        /// consume the record from the beginning.
        void next_record()
        {
            record_.clear();
            position_ = 0;
            bool last_subrecord = false;
            while (!last_subrecord) {
                int32_t head, tail;
                if (!file_.read(reinterpret_cast<char*>(&head), 4)) {
                    throw std::runtime_error("FortranBinaryReader: unexpected end of file in " + filename_);
                }
                // negative head -> more subrecords follow
                last_subrecord = (head >= 0);
                size_t nbytes  = static_cast<size_t>(head >= 0 ? head : -int64_t(head));
                size_t offset  = record_.size();
                record_.resize(offset + nbytes);
                if (nbytes > 0 && !file_.read(record_.data() + offset, nbytes)) {
                    throw std::runtime_error("FortranBinaryReader: truncated record in " + filename_);
                }
                file_.read(reinterpret_cast<char*>(&tail), 4);
                if (!file_ || std::abs(int64_t(tail)) != int64_t(nbytes)) {
                    throw std::runtime_error("FortranBinaryReader: inconsistent record markers in " + filename_ +
                                             " (is it a Fortran unformatted file with 4-byte markers?)");
                }
            }
        }

        size_t record_size() const { return record_.size(); }
        size_t remaining() const { return record_.size() - position_; }

        template <typename T>
        T get()
        {
            T value;
            get(&value, 1);
            return value;
        }

        template <typename T>
        void get(T* ptr, size_t count)
        {
            size_t nbytes = sizeof(T) * count;
            if (position_ + nbytes > record_.size()) {
                throw std::runtime_error("FortranBinaryReader: reading past the end of a record in " + filename_);
            }
            std::memcpy(ptr, record_.data() + position_, nbytes);
            position_ += nbytes;
        }

        /// Fortran LOGICAL (default kind, 4 bytes): any non-zero value is .true.
        bool get_logical() { return get<int32_t>() != 0; }
};

class FortranBinaryWriter
{
    private:
        std::ofstream file_;
        std::vector<char> record_;

    public:
        explicit FortranBinaryWriter(const std::string& filename__)
            : file_(filename__, std::ios::binary)
        {
            if (!file_.is_open()) {
                throw std::runtime_error("FortranBinaryWriter: cannot open file " + filename__);
            }
        }

        template <typename T>
        void put(const T* ptr, size_t count)
        {
            auto offset = record_.size();
            record_.resize(offset + sizeof(T) * count);
            std::memcpy(record_.data() + offset, ptr, sizeof(T) * count);
        }

        template <typename T>
        void put(const T& value) { put(&value, 1); }

        void put_logical(bool value) { int32_t v = value ? 1 : 0; put(v); }

        /// Flush the current record to file (split in sub-records if > 2 GiB).
        void end_record()
        {
            const size_t max_chunk = 2147483639; // same limit used by gfortran
            size_t written = 0;
            do {
                size_t chunk = std::min(max_chunk, record_.size() - written);
                bool last    = (written + chunk == record_.size());
                int32_t head = last ? int32_t(chunk) : -int32_t(chunk);
                int32_t tail = (written == 0) ? int32_t(chunk) : -int32_t(chunk);
                file_.write(reinterpret_cast<char*>(&head), 4);
                file_.write(record_.data() + written, chunk);
                file_.write(reinterpret_cast<char*>(&tail), 4);
                written += chunk;
            } while (written < record_.size());
            record_.clear();
        }
};

#endif
