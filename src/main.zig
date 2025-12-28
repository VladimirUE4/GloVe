const std = @import("std");
const GloVe = @import("GloVe");
const vocab = GloVe.vocab;
const cooccur = GloVe.cooccur;
const glove = GloVe.glove;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    // Skip program name
    _ = args.next();

    const command = args.next() orelse {
        printUsage();
        return;
    };

    if (std.mem.eql(u8, command, "vocab")) {
        try handleVocab(allocator, &args);
    } else if (std.mem.eql(u8, command, "cooccur")) {
        try handleCooccur(allocator, &args);
    } else if (std.mem.eql(u8, command, "train")) {
        try handleTrain(allocator, &args);
    } else if (std.mem.eql(u8, command, "full")) {
        try handleFull(allocator, &args);
    } else {
        printUsage();
    }
}

fn printUsage() void {
    std.debug.print(
        \\GloVe: Global Vectors for Word Representation
        \\
        \\Usage:
        \\  glove vocab <corpus> <output> [--min-count N]
        \\  glove cooccur <corpus> <vocab> <output> [--window-size N] [--symmetric]
        \\  glove train <cooccur> <vocab> <output> [--vector-size N] [--iterations N]
        \\  glove full <corpus> <output> [options...]
        \\
        \\Commands:
        \\  vocab      Build vocabulary from corpus
        \\  cooccur    Build co-occurrence matrix
        \\  train      Train GloVe model
        \\  full       Run full pipeline (vocab -> cooccur -> train)
        \\
        \\Options:
        \\  --min-count N       Minimum word count (default: 5)
        \\  --window-size N     Context window size (default: 15)
        \\  --symmetric         Use symmetric context
        \\  --vector-size N     Vector dimension (default: 50)
        \\  --iterations N      Training iterations (default: 25)
        \\  --x-max N           X_max parameter (default: 100.0)
        \\  --alpha N           Alpha parameter (default: 0.75)
        \\  --learning-rate N   Learning rate (default: 0.05)
        \\
    , .{});
}

fn handleVocab(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    const corpus_path = args.next() orelse {
        std.debug.print("Error: corpus path required\n", .{});
        return;
    };
    const output_path = args.next() orelse {
        std.debug.print("Error: output path required\n", .{});
        return;
    };

    var min_count: u64 = 5;
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--min-count")) {
            const count_str = args.next() orelse {
                std.debug.print("Error: --min-count requires a value\n", .{});
                return;
            };
            min_count = try std.fmt.parseInt(u64, count_str, 10);
        }
    }

    std.debug.print("Building vocabulary from {s}...\n", .{corpus_path});
    var vocabulary = try vocab.buildVocabulary(allocator, corpus_path, min_count);
    defer vocabulary.deinit();

    std.debug.print("Vocabulary size: {d}\n", .{vocabulary.size()});

    // Save vocabulary
    const file = try std.fs.cwd().createFile(output_path, .{});
    defer file.close();
    var output = std.ArrayListUnmanaged(u8){};
    defer output.deinit(allocator);
    const writer = output.writer(allocator);

    for (0..vocabulary.size()) |i| {
        if (vocabulary.getWord(i)) |word| {
            const count = vocabulary.getCount(word) orelse 0;
            try writer.print("{s} {d}\n", .{ word, count });
        }
    }
    try file.writeAll(output.items);

    std.debug.print("Vocabulary saved to {s}\n", .{output_path});
}

fn handleCooccur(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    const corpus_path = args.next() orelse {
        std.debug.print("Error: corpus path required\n", .{});
        return;
    };
    _ = args.next(); // vocab path (not used in this simple version)
    const output_path = args.next() orelse {
        std.debug.print("Error: output path required\n", .{});
        return;
    };

    var window_size: usize = 15;
    var symmetric = false;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--window-size")) {
            const size_str = args.next() orelse {
                std.debug.print("Error: --window-size requires a value\n", .{});
                return;
            };
            window_size = try std.fmt.parseInt(usize, size_str, 10);
        } else if (std.mem.eql(u8, arg, "--symmetric")) {
            symmetric = true;
        }
    }

    std.debug.print("Building co-occurrence matrix...\n", .{});
    // For simplicity, we'll build vocab on the fly
    var vocabulary = try vocab.buildVocabulary(allocator, corpus_path, 1);
    defer vocabulary.deinit();

    var matrix = cooccur.CooccurrenceMatrix.init(allocator, window_size, symmetric);
    defer matrix.deinit();

    try matrix.buildFromCorpus(corpus_path, &vocabulary);
    std.debug.print("Co-occurrence matrix size: {d}\n", .{matrix.size()});

    // Save co-occurrence matrix
    const file = try std.fs.cwd().createFile(output_path, .{});
    defer file.close();
    var output = std.ArrayListUnmanaged(u8){};
    defer output.deinit(allocator);
    const writer = output.writer(allocator);

    for (matrix.cooccurrences.items) |cooc| {
        try writer.print("{d} {d} {d:.6}\n", .{ cooc.word_i, cooc.word_j, cooc.count });
    }
    try file.writeAll(output.items);

    std.debug.print("Co-occurrence matrix saved to {s}\n", .{output_path});
}

fn handleTrain(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    const cooccur_path = args.next() orelse {
        std.debug.print("Error: co-occurrence matrix path required\n", .{});
        return;
    };
    _ = args.next(); // vocab path
    const output_path = args.next() orelse {
        std.debug.print("Error: output path required\n", .{});
        return;
    };

    var vector_size: usize = 50;
    var iterations: usize = 25;
    var x_max: f64 = 100.0;
    var alpha: f64 = 0.75;
    var learning_rate: f64 = 0.05;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--vector-size")) {
            const size_str = args.next() orelse {
                std.debug.print("Error: --vector-size requires a value\n", .{});
                return;
            };
            vector_size = try std.fmt.parseInt(usize, size_str, 10);
        } else if (std.mem.eql(u8, arg, "--iterations")) {
            const iter_str = args.next() orelse {
                std.debug.print("Error: --iterations requires a value\n", .{});
                return;
            };
            iterations = try std.fmt.parseInt(usize, iter_str, 10);
        } else if (std.mem.eql(u8, arg, "--x-max")) {
            const xmax_str = args.next() orelse {
                std.debug.print("Error: --x-max requires a value\n", .{});
                return;
            };
            x_max = try std.fmt.parseFloat(f64, xmax_str);
        } else if (std.mem.eql(u8, arg, "--alpha")) {
            const alpha_str = args.next() orelse {
                std.debug.print("Error: --alpha requires a value\n", .{});
                return;
            };
            alpha = try std.fmt.parseFloat(f64, alpha_str);
        } else if (std.mem.eql(u8, arg, "--learning-rate")) {
            const lr_str = args.next() orelse {
                std.debug.print("Error: --learning-rate requires a value\n", .{});
                return;
            };
            learning_rate = try std.fmt.parseFloat(f64, lr_str);
        }
    }

    // Load co-occurrence matrix
    const file = try std.fs.cwd().openFile(cooccur_path, .{});
    defer file.close();

    const chunk_size = 1024 * 1024; // 1MB chunk
    const buffer = try allocator.alloc(u8, chunk_size);
    defer allocator.free(buffer);

    var left_over = std.ArrayListUnmanaged(u8){};
    defer left_over.deinit(allocator);

    var matrix = cooccur.CooccurrenceMatrix.init(allocator, 15, true);
    defer matrix.deinit();
    var max_word_idx: usize = 0;

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
                try processCooccurLine(left_over.items, &matrix, &max_word_idx);
                left_over.clearRetainingCapacity();
            } else {
                try processCooccurLine(line_part, &matrix, &max_word_idx);
            }
            start = end + 1;
        }

        if (start < chunk.len) {
            try left_over.appendSlice(allocator, chunk[start..]);
        }
    }

    if (left_over.items.len > 0) {
        try processCooccurLine(left_over.items, &matrix, &max_word_idx);
    }

    const vocab_size = max_word_idx + 1;

    std.debug.print("Training GloVe model (vocab_size={d}, vector_size={d})...\n", .{ vocab_size, vector_size });

    var model = try glove.GloVeModel.init(allocator, vocab_size, vector_size, x_max, alpha, learning_rate);
    defer model.deinit();

    try model.train(&matrix, iterations);

    // Create a dummy vocabulary for saving (in real implementation, load from file)
    var vocabulary = vocab.Vocabulary.init(allocator, 1);
    defer vocabulary.deinit();
    for (0..vocab_size) |i| {
        var word_buf: [32]u8 = undefined;
        const word = try std.fmt.bufPrint(&word_buf, "word_{d}", .{i});
        try vocabulary.addWord(word);
    }
    try vocabulary.finalize();

    try model.saveVectors(&vocabulary, output_path);
    std.debug.print("Vectors saved to {s}\n", .{output_path});
}

fn processCooccurLine(line: []const u8, matrix: *cooccur.CooccurrenceMatrix, max_word_idx: *usize) !void {
    var parts = std.mem.tokenizeScalar(u8, line, ' ');
    const word_i_str = parts.next() orelse return;
    const word_j_str = parts.next() orelse return;
    const count_str = parts.next() orelse return;

    const word_i = try std.fmt.parseInt(usize, word_i_str, 10);
    const word_j = try std.fmt.parseInt(usize, word_j_str, 10);
    const count = try std.fmt.parseFloat(f64, count_str);

    max_word_idx.* = @max(max_word_idx.*, @max(word_i, word_j));
    try matrix.addCooccurrence(word_i, word_j, count);
}

fn handleFull(allocator: std.mem.Allocator, args: *std.process.ArgIterator) !void {
    const corpus_path = args.next() orelse {
        std.debug.print("Error: corpus path required\n", .{});
        return;
    };
    const output_path = args.next() orelse {
        std.debug.print("Error: output path required\n", .{});
        return;
    };

    var min_count: u64 = 5;
    var window_size: usize = 15;
    var vector_size: usize = 50;
    var iterations: usize = 25;
    var x_max: f64 = 100.0;
    var alpha: f64 = 0.75;
    var learning_rate: f64 = 0.05;
    var symmetric = true;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--min-count")) {
            const count_str = args.next() orelse continue;
            min_count = std.fmt.parseInt(u64, count_str, 10) catch min_count;
        } else if (std.mem.eql(u8, arg, "--window-size")) {
            const size_str = args.next() orelse continue;
            window_size = std.fmt.parseInt(usize, size_str, 10) catch window_size;
        } else if (std.mem.eql(u8, arg, "--vector-size")) {
            const size_str = args.next() orelse continue;
            vector_size = std.fmt.parseInt(usize, size_str, 10) catch vector_size;
        } else if (std.mem.eql(u8, arg, "--iterations")) {
            const iter_str = args.next() orelse continue;
            iterations = std.fmt.parseInt(usize, iter_str, 10) catch iterations;
        } else if (std.mem.eql(u8, arg, "--x-max")) {
            const xmax_str = args.next() orelse continue;
            x_max = std.fmt.parseFloat(f64, xmax_str) catch x_max;
        } else if (std.mem.eql(u8, arg, "--alpha")) {
            const alpha_str = args.next() orelse continue;
            alpha = std.fmt.parseFloat(f64, alpha_str) catch alpha;
        } else if (std.mem.eql(u8, arg, "--learning-rate")) {
            const lr_str = args.next() orelse continue;
            learning_rate = std.fmt.parseFloat(f64, lr_str) catch learning_rate;
        } else if (std.mem.eql(u8, arg, "--symmetric")) {
            symmetric = true;
        }
    }

    std.debug.print("Running full GloVe pipeline...\n", .{});

    // Step 1: Build vocabulary
    std.debug.print("Step 1: Building vocabulary...\n", .{});
    var vocabulary = try vocab.buildVocabulary(allocator, corpus_path, min_count);
    defer vocabulary.deinit();
    std.debug.print("Vocabulary size: {d}\n", .{vocabulary.size()});

    // Step 2: Build co-occurrence matrix
    std.debug.print("Step 2: Building co-occurrence matrix...\n", .{});
    var matrix = cooccur.CooccurrenceMatrix.init(allocator, window_size, symmetric);
    defer matrix.deinit();
    try matrix.buildFromCorpus(corpus_path, &vocabulary);
    std.debug.print("Co-occurrence matrix size: {d}\n", .{matrix.size()});

    // Step 3: Train model
    std.debug.print("Step 3: Training GloVe model...\n", .{});
    var model = try glove.GloVeModel.init(allocator, vocabulary.size(), vector_size, x_max, alpha, learning_rate);
    defer model.deinit();
    try model.train(&matrix, iterations);

    // Step 4: Save vectors
    std.debug.print("Step 4: Saving vectors...\n", .{});
    try model.saveVectors(&vocabulary, output_path);
    std.debug.print("Done! Vectors saved to {s}\n", .{output_path});
}

test "simple test" {
    const gpa = std.testing.allocator;
    var list: std.ArrayListUnmanaged(i32) = .{};
    defer list.deinit(gpa);
    try list.append(gpa, 42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
