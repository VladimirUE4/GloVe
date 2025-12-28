//! GloVe: Global Vectors for Word Representation
//! A Zig implementation of the GloVe algorithm for learning word embeddings
const std = @import("std");

// Re-export all modules
pub const vocab = @import("vocab.zig");
pub const cooccur = @import("cooccur.zig");
pub const glove = @import("glove.zig");

// Re-export types for convenience
pub const Vocabulary = vocab.Vocabulary;
pub const CooccurrenceMatrix = cooccur.CooccurrenceMatrix;
pub const GloVeModel = glove.GloVeModel;

pub fn bufferedPrint() !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("GloVe: Global Vectors for Word Representation\n", .{});
    try stdout.print("Zig implementation\n", .{});

    try stdout.flush();
}

pub fn add(a: i32, b: i32) i32 {
    return a + b;
}

test "basic add functionality" {
    try std.testing.expect(add(3, 7) == 10);
}
