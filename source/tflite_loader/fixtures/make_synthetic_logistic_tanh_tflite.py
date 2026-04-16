#!/usr/bin/env python3
import argparse
import flatbuffers


def i32_vector(builder, values):
    builder.StartVector(4, len(values), 4)
    for value in reversed(values):
        builder.PrependInt32(value)
    return builder.EndVector()


def table_vector(builder, offsets):
    builder.StartVector(4, len(offsets), 4)
    for off in reversed(offsets):
        builder.PrependUOffsetTRelative(off)
    return builder.EndVector()


def make_tensor(builder, name, shape, buffer_index):
    name_off = builder.CreateString(name)
    shape_off = i32_vector(builder, shape)
    builder.StartObject(16)
    builder.PrependUOffsetTRelativeSlot(0, shape_off, 0)
    builder.PrependInt8Slot(1, 0, 0)
    builder.PrependUint32Slot(2, buffer_index, 0)
    builder.PrependUOffsetTRelativeSlot(3, name_off, 0)
    return builder.EndObject()


def make_operator_code(builder, builtin_code):
    builder.StartObject(4)
    builder.PrependInt8Slot(0, builtin_code, 0)
    builder.PrependInt32Slot(3, builtin_code, 0)
    builder.PrependInt32Slot(2, 1, 0)
    return builder.EndObject()


def make_operator(builder, opcode_index, inputs, outputs):
    inputs_off = i32_vector(builder, inputs)
    outputs_off = i32_vector(builder, outputs)
    builder.StartObject(6)
    builder.PrependUint32Slot(0, opcode_index, 0)
    builder.PrependUOffsetTRelativeSlot(1, inputs_off, 0)
    builder.PrependUOffsetTRelativeSlot(2, outputs_off, 0)
    return builder.EndObject()


def make_buffer(builder):
    builder.StartObject(1)
    return builder.EndObject()


def build_model():
    builder = flatbuffers.Builder(2048)
    desc = builder.CreateString("synthetic float32 logistic+tanh model for generic scheduler smoke test")
    input_tensor = make_tensor(builder, "input", [1, 8], 0)
    mid_tensor = make_tensor(builder, "logistic_out", [1, 8], 1)
    output_tensor = make_tensor(builder, "tanh_out", [1, 8], 2)
    tensors = table_vector(builder, [input_tensor, mid_tensor, output_tensor])
    inputs = i32_vector(builder, [0])
    outputs = i32_vector(builder, [2])
    op0 = make_operator(builder, 0, [0], [1])
    op1 = make_operator(builder, 1, [1], [2])
    operators = table_vector(builder, [op0, op1])
    name = builder.CreateString("main")
    builder.StartObject(5)
    builder.PrependUOffsetTRelativeSlot(0, tensors, 0)
    builder.PrependUOffsetTRelativeSlot(1, inputs, 0)
    builder.PrependUOffsetTRelativeSlot(2, outputs, 0)
    builder.PrependUOffsetTRelativeSlot(3, operators, 0)
    builder.PrependUOffsetTRelativeSlot(4, name, 0)
    subgraph = builder.EndObject()
    opcodes = table_vector(builder, [make_operator_code(builder, 14), make_operator_code(builder, 28)])
    subgraphs = table_vector(builder, [subgraph])
    buffers = table_vector(builder, [make_buffer(builder), make_buffer(builder), make_buffer(builder)])
    builder.StartObject(8)
    builder.PrependInt32Slot(0, 3, 0)
    builder.PrependUOffsetTRelativeSlot(1, opcodes, 0)
    builder.PrependUOffsetTRelativeSlot(2, subgraphs, 0)
    builder.PrependUOffsetTRelativeSlot(3, desc, 0)
    builder.PrependUOffsetTRelativeSlot(4, buffers, 0)
    model = builder.EndObject()
    builder.Finish(model, file_identifier=b"TFL3")
    return bytes(builder.Output())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(args.out, "wb") as f:
        f.write(build_model())


if __name__ == "__main__":
    main()
