//! Vocabulary building and word counting
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const WordInfo = struct {
    word: []const u8,
    count: u64,
};

pub const Vocabulary = struct {
    allocator: Allocator,
    words: std.StringHashMap(u64),
    sorted_words: std.ArrayListUnmanaged([]const u8),
    min_count: u64,

    pub fn init(allocator: Allocator, min_count: u64) Vocabulary {
        return Vocabulary{
            .allocator = allocator,
            .words = std.StringHashMap(u64).init(allocator),
            .sorted_words = .{},
            .min_count = min_count,
        };
    }

    pub fn deinit(self: *Vocabulary) void {
        var it = self.words.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.words.deinit();
        self.sorted_words.deinit(self.allocator);
    }

    pub fn addWord(self: *Vocabulary, word: []const u8) !void {
        const owned_word = try self.allocator.dupe(u8, word);
        errdefer self.allocator.free(owned_word);

        const gop = try self.words.getOrPut(owned_word);
        if (gop.found_existing) {
            gop.value_ptr.* += 1;
            self.allocator.free(owned_word);
        } else {
            gop.value_ptr.* = 1;
        }
    }

    pub fn finalize(self: *Vocabulary) !void {
        // Filter words by min_count and sort
        var to_remove = std.ArrayListUnmanaged([]const u8){};
        defer to_remove.deinit(self.allocator);

        var it = self.words.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.* >= self.min_count) {
                try self.sorted_words.append(self.allocator, entry.key_ptr.*);
            } else {
                try to_remove.append(self.allocator, entry.key_ptr.*);
            }
        }

        for (to_remove.items) |key| {
            _ = self.words.remove(key);
            self.allocator.free(key);
        }

        // Sort by frequency (descending)
        const Context = struct {
            vocab: *const Vocabulary,
        };
        const ctx = Context{ .vocab = self };
        std.mem.sort([]const u8, self.sorted_words.items, ctx, struct {
            fn lessThan(context: Context, a: []const u8, b: []const u8) bool {
                const count_a = context.vocab.words.get(a) orelse 0;
                const count_b = context.vocab.words.get(b) orelse 0;
                if (count_a != count_b) {
                    return count_a > count_b;
                }
                return std.mem.order(u8, a, b) == .lt;
            }
        }.lessThan);
    }

    pub fn getCount(self: *const Vocabulary, word: []const u8) ?u64 {
        return self.words.get(word);
    }

    pub fn getIndex(self: *const Vocabulary, word: []const u8) ?usize {
        for (self.sorted_words.items, 0..) |w, i| {
            if (std.mem.eql(u8, w, word)) {
                return i;
            }
        }
        return null;
    }

    pub fn size(self: *const Vocabulary) usize {
        return self.sorted_words.items.len;
    }

    pub fn getWord(self: *const Vocabulary, index: usize) ?[]const u8 {
        if (index >= self.sorted_words.items.len) return null;
        return self.sorted_words.items[index];
    }
};

pub fn buildVocabulary(allocator: Allocator, corpus_path: []const u8, min_count: u64) !Vocabulary {
    var vocab = Vocabulary.init(allocator, min_count);
    errdefer vocab.deinit();

    const file = try std.fs.cwd().openFile(corpus_path, .{});
    defer file.close();

    const chunk_size = 1024 * 1024; // 1MB chunk
    const buffer = try allocator.alloc(u8, chunk_size);
    defer allocator.free(buffer);

    var left_over = std.ArrayListUnmanaged(u8){};
    defer left_over.deinit(allocator);

    while (true) {
        const bytes_read = try file.read(buffer);
        if (bytes_read == 0) break;

        const chunk = buffer[0..bytes_read];
        var start: usize = 0;

        while (std.mem.indexOfScalar(u8, chunk[start..], '\n')) |pos| {
            const end = start + pos;
            const line_part = chunk[start..end];

            if (left_over.items.len > 0) {
                try left_over.appendSlice(allocator, line_part);
                try processLine(&vocab, left_over.items);
                left_over.clearRetainingCapacity();
            } else {
                try processLine(&vocab, line_part);
            }
            start = end + 1;
        }

        if (start < chunk.len) {
            try left_over.appendSlice(allocator, chunk[start..]);
        }
    }

    if (left_over.items.len > 0) {
        try processLine(&vocab, left_over.items);
    }

    try vocab.finalize();
    return vocab;
}

fn processLine(vocab: *Vocabulary, line: []const u8) !void {
    var iter = std.mem.tokenizeScalar(u8, line, ' ');
    while (iter.next()) |token| {
        if (token.len > 0) {
            try vocab.addWord(token);
        }
    }
}

test "vocabulary building" {
    const gpa = std.testing.allocator;
    var vocab = try Vocabulary.init(gpa, 1);
    defer vocab.deinit();

    try vocab.addWord("hello");
    try vocab.addWord("world");
    try vocab.addWord("hello");
    try vocab.finalize();

    try std.testing.expect(vocab.size() == 2);
    try std.testing.expect(vocab.getCount("hello").? == 2);
    try std.testing.expect(vocab.getCount("world").? == 1);
}
