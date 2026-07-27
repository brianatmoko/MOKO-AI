#ifndef MOKO_MMAP_IO_HPP
#define MOKO_MMAP_IO_HPP

#include <string>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstdint>
#include <iostream>

namespace moko {

class MemoryMappedFile {
private:
    int fd;
    size_t length;
    uint8_t* data_ptr;
    bool valid;

public:
    MemoryMappedFile(const std::string& filepath) : fd(-1), length(0), data_ptr(nullptr), valid(false) {
        fd = open(filepath.c_str(), O_RDONLY);
        if (fd == -1) {
            // File might not exist or permission denied
            return;
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            return;
        }
        
        length = sb.st_size;
        if (length == 0) {
            close(fd);
            return; // Empty file
        }

        data_ptr = static_cast<uint8_t*>(mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0));
        if (data_ptr == MAP_FAILED) {
            data_ptr = nullptr;
            close(fd);
            return;
        }
        
        valid = true;
        
        // ── OPTIMISASI KERNEL HINT ─────────────────────────────────────────
        // MADV_SEQUENTIAL: beritahu kernel data dibaca dari awal ke akhir.
        //   Kernel akan melakukan read-ahead agresif, mengurangi page-fault
        //   dan idle CPU saat menunggu I/O disk.
        // MADV_WILLNEED: mulai prefetch segera ke page cache RAM.
        //   Menghilangkan cold-miss saat scan vektor pertama dimulai.
        // !! SEBELUMNYA MADV_RANDOM — justru mematikan read-ahead kernel !!
        madvise(data_ptr, length, MADV_SEQUENTIAL);
        madvise(data_ptr, length, MADV_WILLNEED);
        
        // MADV_HUGEPAGE: gunakan 2MB transparent hugepages untuk file besar.
        // Mengurangi TLB pressure saat melintasi jutaan byte vektor.
        if (length >= (1024 * 1024)) {
            madvise(data_ptr, length, MADV_HUGEPAGE);
        }
    }

    ~MemoryMappedFile() {
        if (data_ptr && data_ptr != MAP_FAILED) {
            munmap(data_ptr, length);
        }
        if (fd != -1) {
            close(fd);
        }
    }

    bool isValid() const { return valid; }
    size_t size() const { return length; }
    const uint8_t* data() const { return data_ptr; }
};

} // namespace moko

#endif // MOKO_MMAP_IO_HPP
