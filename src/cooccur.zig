//! Co-occurrence matrix construction
const std = @import("std");
const Allocator = std.mem.Allocator;
const vocab = @import("vocab.zig");

pub const Cooccurrence = struct {
    word_i: usize,
    word_j: usize,
    count: f64,
};

pub const CooccurrenceMatrix = struct {
    allocator: Allocator,
    cooccurrences: std.ArrayListUnmanaged(Cooccurrence),
    window_size: usize,
    symmetric: bool,

    pub fn init(allocator: Allocator, window_size: usize, symmetric: bool) CooccurrenceMatrix {
        return CooccurrenceMatrix{
            .allocator = allocator,
            .cooccurrences = .{},
            .window_size = window_size,
            .symmetric = symmetric,
        };
    }

    pub fn deinit(self: *CooccurrenceMatrix) void {
        self.cooccurrences.deinit(self.allocator);
    }

    pub fn addCooccurrence(self: *CooccurrenceMatrix, word_i: usize, word_j: usize, count: f64) !void {
        try self.cooccurrences.append(self.allocator, Cooccurrence{
            .word_i = word_i,
            .word_j = word_j,
            .count = count,
        });
    }

    pub fn buildFromCorpus(
        self: *CooccurrenceMatrix,
        corpus_path: []const u8,
        vocabulary: *vocab.Vocabulary,
    ) !void {
        const file = try std.fs.cwd().openFile(corpus_path, .{});
        defer file.close();

        const chunk_size = 1024 * 1024; // 1MB chunk
        const buffer = try self.allocator.alloc(u8, chunk_size);
        defer self.allocator.free(buffer);

        var left_over = std.ArrayListUnmanaged(u8){};
        defer left_over.deinit(self.allocator);

        var sentence = std.ArrayListUnmanaged(usize){};
        defer sentence.deinit(self.allocator);

        while (true) {
            const bytes_read = try file.read(buffer);
            if (bytes_read == 0) break;

            const chunk = buffer[0..bytes_read];
            var start: usize = 0;

            while (std.mem.indexOfScalar(u8, chunk[start..], '\n')) |pos| {
                const end = start + pos;
                const line_part = chunk[start..end];

                if (left_over.items.len > 0) {
                    try left_over.appendSlice(self.allocator, line_part);
                    try self.processLine(left_over.items, vocabulary, &sentence);
                    left_over.clearRetainingCapacity();
                } else {
                    try self.processLine(line_part, vocabulary, &sentence);
                }
                start = end + 1;
            }

            if (start < chunk.len) {
                try left_over.appendSlice(self.allocator, chunk[start..]);
            }
        }

        if (left_over.items.len > 0) {
            try self.processLine(left_over.items, vocabulary, &sentence);
        }
    }

    fn processLine(self: *CooccurrenceMatrix, line: []const u8, vocabulary: *vocab.Vocabulary, sentence: *std.ArrayListUnmanaged(usize)) !void {
        sentence.clearRetainingCapacity();

        // Convert line to word indices
        var iter = std.mem.tokenizeScalar(u8, line, ' ');
        while (iter.next()) |token| {
            if (vocabulary.getIndex(token)) |idx| {
                try sentence.append(self.allocator, idx);
            }
        }

        // Build co-occurrences for this sentence
        for (sentence.items, 0..) |word_i, i| {
            const start = if (i >= self.window_size) i - self.window_size else 0;
            const end = @min(i + self.window_size + 1, sentence.items.len);

            var j = start;
            while (j < end) : (j += 1) {
                if (j == i) continue;
                const word_j = sentence.items[j];
                const distance = @as(f64, @floatFromInt(if (i > j) i - j else j - i));
                const weight = 1.0 / distance;

                try self.addCooccurrence(word_i, word_j, weight);
                if (self.symmetric) {
                    try self.addCooccurrence(word_j, word_i, weight);
                }
            }
        }
    }

    pub fn size(self: *const CooccurrenceMatrix) usize {
        return self.cooccurrences.items.len;
    }
};

test "cooccurrence matrix" {
    const gpa = std.testing.allocator;
    var vocab_instance = try vocab.Vocabulary.init(gpa, 1);
    defer vocab_instance.deinit();

    try vocab_instance.addWord("hello");
    try vocab_instance.addWord("world");
    try vocab_instance.finalize();

    var matrix = CooccurrenceMatrix.init(gpa, 2, true);
    defer matrix.deinit();

    try std.testing.expect(matrix.window_size == 2);
}
