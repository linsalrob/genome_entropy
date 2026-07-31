#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static void usage(const char *program)
{
    fprintf(stderr, "Usage: %s GENBANK_FILE LOCUS_ID OUTPUT_FILE\n", program);
}

static int write_all(int fd, const char *data, size_t length)
{
    while (length > 0) {
        ssize_t written = write(fd, data, length);

        if (written < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        data += (size_t)written;
        length -= (size_t)written;
    }
    return 0;
}

static const char *line_end(const char *start, const char *end)
{
    const char *newline = memchr(start, '\n', (size_t)(end - start));
    return newline == NULL ? end : newline;
}

static const char *next_line(const char *end_of_line, const char *end)
{
    return end_of_line < end ? end_of_line + 1 : end;
}

static size_t content_length(const char *start, const char *end_of_line)
{
    size_t length = (size_t)(end_of_line - start);
    return length > 0 && start[length - 1] == '\r' ? length - 1 : length;
}

static int locus_matches(const char *start, const char *end_of_line,
                         const char *wanted, size_t wanted_length)
{
    size_t length = content_length(start, end_of_line);
    const char *cursor;
    const char *limit = start + length;
    const char *id_start;

    if (length <= 5 || memcmp(start, "LOCUS", 5) != 0)
        return 0;
    cursor = start + 5;
    if (*cursor != ' ' && *cursor != '\t')
        return 0;
    while (cursor < limit && (*cursor == ' ' || *cursor == '\t'))
        ++cursor;
    id_start = cursor;
    while (cursor < limit && *cursor != ' ' && *cursor != '\t')
        ++cursor;

    return (size_t)(cursor - id_start) == wanted_length &&
           memcmp(id_start, wanted, wanted_length) == 0;
}

static const char *record_end(const char *start, const char *end)
{
    const char *cursor = start;

    while (cursor < end) {
        const char *end_of_line = line_end(cursor, end);
        size_t length = content_length(cursor, end_of_line);

        if (length == 2 && cursor[0] == '/' && cursor[1] == '/')
            return next_line(end_of_line, end);
        cursor = next_line(end_of_line, end);
    }
    return NULL;
}

int main(int argc, char **argv)
{
    int input_fd = -1;
    int output_fd = -1;
    struct stat input_stat;
    struct stat output_stat;
    const char *mapping = MAP_FAILED;
    const char *end;
    const char *cursor;
    const char *match = NULL;
    const char *match_end = NULL;
    size_t locus_length;
    int result = EXIT_FAILURE;

    if (argc != 4) {
        usage(argv[0]);
        return 2;
    }
    locus_length = strlen(argv[2]);
    if (locus_length == 0 || strpbrk(argv[2], " \t\r\n") != NULL) {
        fprintf(stderr, "LOCUS_ID must be one non-empty, whitespace-free token\n");
        return 2;
    }

    input_fd = open(argv[1], O_RDONLY);
    if (input_fd < 0) {
        fprintf(stderr, "Cannot open input '%s': %s\n", argv[1], strerror(errno));
        goto cleanup;
    }
    if (fstat(input_fd, &input_stat) < 0) {
        fprintf(stderr, "Cannot inspect input '%s': %s\n", argv[1], strerror(errno));
        goto cleanup;
    }
    if (!S_ISREG(input_stat.st_mode)) {
        fprintf(stderr, "Input '%s' is not a regular file\n", argv[1]);
        goto cleanup;
    }
    if (input_stat.st_size == 0) {
        fprintf(stderr, "LOCUS '%s' was not found in '%s'\n", argv[2], argv[1]);
        result = 3;
        goto cleanup;
    }

    mapping = mmap(NULL, (size_t)input_stat.st_size, PROT_READ, MAP_PRIVATE,
                   input_fd, 0);
    if (mapping == MAP_FAILED) {
        fprintf(stderr, "Cannot map input '%s': %s\n", argv[1], strerror(errno));
        goto cleanup;
    }
#ifdef POSIX_MADV_SEQUENTIAL
    (void)posix_madvise((void *)mapping, (size_t)input_stat.st_size,
                        POSIX_MADV_SEQUENTIAL);
#endif

    end = mapping + input_stat.st_size;
    cursor = mapping;
    while (cursor < end) {
        const char *end_of_line = line_end(cursor, end);

        if (locus_matches(cursor, end_of_line, argv[2], locus_length)) {
            match = cursor;
            match_end = record_end(next_line(end_of_line, end), end);
            break;
        }
        cursor = next_line(end_of_line, end);
    }

    if (match == NULL) {
        fprintf(stderr, "LOCUS '%s' was not found in '%s'\n", argv[2], argv[1]);
        result = 3;
        goto cleanup;
    }
    if (match_end == NULL) {
        fprintf(stderr, "LOCUS '%s' has no terminating // line\n", argv[2]);
        result = 4;
        goto cleanup;
    }

    output_fd = open(argv[3], O_WRONLY | O_CREAT, 0666);
    if (output_fd < 0) {
        fprintf(stderr, "Cannot open output '%s': %s\n", argv[3], strerror(errno));
        goto cleanup;
    }
    if (fstat(output_fd, &output_stat) < 0) {
        fprintf(stderr, "Cannot inspect output '%s': %s\n", argv[3], strerror(errno));
        goto cleanup;
    }
    if (input_stat.st_dev == output_stat.st_dev &&
        input_stat.st_ino == output_stat.st_ino) {
        fprintf(stderr, "Input and output must be different files\n");
        goto cleanup;
    }
    if (ftruncate(output_fd, 0) < 0 ||
        write_all(output_fd, match, (size_t)(match_end - match)) < 0) {
        fprintf(stderr, "Cannot write output '%s': %s\n", argv[3], strerror(errno));
        goto cleanup;
    }

    result = EXIT_SUCCESS;

cleanup:
    if (output_fd >= 0 && close(output_fd) < 0 && result == EXIT_SUCCESS) {
        fprintf(stderr, "Cannot close output '%s': %s\n", argv[3], strerror(errno));
        result = EXIT_FAILURE;
    }
    if (mapping != MAP_FAILED)
        (void)munmap((void *)mapping, (size_t)input_stat.st_size);
    if (input_fd >= 0)
        (void)close(input_fd);
    return result;
}
