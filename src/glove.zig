//! GloVe training algorithm
const std = @import("std");
const Allocator = std.mem.Allocator;
const vocab = @import("vocab.zig");
const cooccur = @import("cooccur.zig");
const math = std.math;

pub const GloVeModel = struct {
    allocator: Allocator,
    vocab_size: usize,
    vector_size: usize,
    w: std.ArrayListUnmanaged(std.ArrayListUnmanaged(f64)), // Word vectors
    w_hat: std.ArrayListUnmanaged(std.ArrayListUnmanaged(f64)), // Context vectors
    b: std.ArrayListUnmanaged(f64), // Word biases
    b_hat: std.ArrayListUnmanaged(f64), // Context biases
    x_max: f64,
    alpha: f64,
    learning_rate: f64,

    pub fn init(allocator: Allocator, vocab_size: usize, vector_size: usize, x_max: f64, alpha: f64, learning_rate: f64) !GloVeModel {
        var w = std.ArrayListUnmanaged(std.ArrayListUnmanaged(f64)){};
        var w_hat = std.ArrayListUnmanaged(std.ArrayListUnmanaged(f64)){};
        var b = std.ArrayListUnmanaged(f64){};
        var b_hat = std.ArrayListUnmanaged(f64){};

        // Initialize word vectors and context vectors
        var prng = std.Random.DefaultPrng.init(0);
        const random = prng.random();

        var i: usize = 0;
        while (i < vocab_size) : (i += 1) {
            var vec = std.ArrayListUnmanaged(f64){};
            var vec_hat = std.ArrayListUnmanaged(f64){};

            var j: usize = 0;
            while (j < vector_size) : (j += 1) {
                // Initialize with small random values
                const val = (random.float(f64) - 0.5) * 0.01;
                try vec.append(allocator, val);
                try vec_hat.append(allocator, val);
            }

            try w.append(allocator, vec);
            try w_hat.append(allocator, vec_hat);
            try b.append(allocator, 0.0);
            try b_hat.append(allocator, 0.0);
        }

        return GloVeModel{
            .allocator = allocator,
            .vocab_size = vocab_size,
            .vector_size = vector_size,
            .w = w,
            .w_hat = w_hat,
            .b = b,
            .b_hat = b_hat,
            .x_max = x_max,
            .alpha = alpha,
            .learning_rate = learning_rate,
        };
    }

    pub fn deinit(self: *GloVeModel) void {
        for (self.w.items) |*vec| {
            vec.deinit(self.allocator);
        }
        self.w.deinit(self.allocator);

        for (self.w_hat.items) |*vec| {
            vec.deinit(self.allocator);
        }
        self.w_hat.deinit(self.allocator);

        self.b.deinit(self.allocator);
        self.b_hat.deinit(self.allocator);
    }

    // Weighting function f(x) = (x/x_max)^alpha if x < x_max, else 1
    fn weight(self: *const GloVeModel, x: f64) f64 {
        if (x >= self.x_max) {
            return 1.0;
        }
        return math.pow(f64, x / self.x_max, self.alpha);
    }

    // Compute dot product of two vectors
    fn dotProduct(vec1: []const f64, vec2: []const f64) f64 {
        var sum: f64 = 0.0;
        for (vec1, vec2) |v1, v2| {
            sum += v1 * v2;
        }
        return sum;
    }

    // Train on a single co-occurrence
    pub fn trainOnCooccurrence(self: *GloVeModel, word_i: usize, word_j: usize, x_ij: f64) void {
        if (word_i >= self.vocab_size or word_j >= self.vocab_size) return;

        const w_i = self.w.items[word_i].items;
        const w_j_hat = self.w_hat.items[word_j].items;
        const b_i = self.b.items[word_i];
        const b_j_hat = self.b_hat.items[word_j];

        // Compute prediction
        const inner = dotProduct(w_i, w_j_hat) + b_i + b_j_hat;
        const diff = inner - @log(x_ij + 1e-8);

        // Compute weight
        const f_ij = self.weight(x_ij);

        // Compute gradient scale
        const scale = f_ij * diff * self.learning_rate;

        // Update word vectors
        for (w_i, w_j_hat) |*w_i_val, w_j_hat_val| {
            const grad_i = scale * w_j_hat_val;
            w_i_val.* -= grad_i;
        }

        // Update context vector
        const w_j_hat_vec = self.w_hat.items[word_j].items;
        for (w_j_hat_vec, w_i) |*w_j_hat_val, w_i_val| {
            const grad_j = scale * w_i_val;
            w_j_hat_val.* -= grad_j;
        }

        // Update biases
        self.b.items[word_i] -= scale;
        self.b_hat.items[word_j] -= scale;
    }

    // Train on co-occurrence matrix
    pub fn train(self: *GloVeModel, matrix: *cooccur.CooccurrenceMatrix, num_iterations: usize) !void {
        std.debug.print("Training GloVe model on {d} co-occurrences...\n", .{matrix.size()});

        var iteration: usize = 0;
        while (iteration < num_iterations) : (iteration += 1) {
            std.debug.print("Iteration {d}/{d}\n", .{ iteration + 1, num_iterations });

            // Shuffle co-occurrences
            var prng = std.Random.DefaultPrng.init(@intCast(iteration));
            const random = prng.random();
            random.shuffle(cooccur.Cooccurrence, matrix.cooccurrences.items);

            // Train on each co-occurrence
            for (matrix.cooccurrences.items) |cooc| {
                self.trainOnCooccurrence(cooc.word_i, cooc.word_j, cooc.count);
            }
        }
    }

    // Get final word vector (average of w and w_hat)
    pub fn getWordVector(self: *const GloVeModel, word_idx: usize) ![]f64 {
        if (word_idx >= self.vocab_size) {
            return error.InvalidIndex;
        }

        const result = try self.allocator.alloc(f64, self.vector_size);
        const w_vec = self.w.items[word_idx].items;
        const w_hat_vec = self.w_hat.items[word_idx].items;

        for (result, w_vec, w_hat_vec) |*r, w_val, w_hat_val| {
            r.* = (w_val + w_hat_val) / 2.0;
        }

        return result;
    }

    // Save vectors to file
    pub fn saveVectors(self: *const GloVeModel, vocabulary: *vocab.Vocabulary, output_path: []const u8) !void {
        const file = try std.fs.cwd().createFile(output_path, .{});
        defer file.close();

        var output = std.ArrayListUnmanaged(u8){};
        defer output.deinit(self.allocator);
        const writer = output.writer(self.allocator);

        // Write header
        try writer.print("{d} {d}\n", .{ self.vocab_size, self.vector_size });

        // Write vectors
        for (0..self.vocab_size) |i| {
            if (vocabulary.getWord(i)) |word| {
                const vec = try self.getWordVector(i);
                defer self.allocator.free(vec);

                try writer.print("{s}", .{word});
                for (vec) |val| {
                    try writer.print(" {d:.6}", .{val});
                }
                try writer.print("\n", .{});
            }
        }
        try file.writeAll(output.items);
    }
};

test "glove model initialization" {
    const gpa = std.testing.allocator;
    var model = try GloVeModel.init(gpa, 10, 50, 100.0, 0.75, 0.05);
    defer model.deinit();

    try std.testing.expect(model.vocab_size == 10);
    try std.testing.expect(model.vector_size == 50);
    try std.testing.expect(model.w.items.len == 10);
    try std.testing.expect(model.w.items[0].items.len == 50);
}

