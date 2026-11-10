// Const blobs whose stored data is shorter, missing, or whose external length exceeds the tensor must fail to load.
#include <MNN/expr/Module.hpp>
#include "MNNTestSuite.h"
#include "MNN_generated.h"
#include <cstdio>
#include <memory>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#ifdef __ANDROID__
#define MNN_BLOB_BOUNDS_TMP "/data/local/tmp/"
#else
#define MNN_BLOB_BOUNDS_TMP "/tmp/"
#endif

using namespace MNN;
using namespace MNN::Express;

enum class BlobKind { InlineShort, InlineMissing, ExternalTooLong, HalfExternalTooLong, HalfInlineShort, Valid };

static std::vector<uint8_t> constGraph(BlobKind kind, int64_t externalLength) {
    flatbuffers::FlatBufferBuilder builder;
    const auto dims = builder.CreateVector(std::vector<int>{4});
    flatbuffers::Offset<flatbuffers::Vector<float>> floats;
    flatbuffers::Offset<flatbuffers::Vector<uint8_t>> bytes;
    flatbuffers::Offset<flatbuffers::Vector<int64_t>> external;
    DataType type = DataType_DT_FLOAT;
    switch (kind) {
        case BlobKind::InlineShort:
            floats = builder.CreateVector(std::vector<float>{1.f});
            break;
        case BlobKind::InlineMissing:
            break;
        case BlobKind::ExternalTooLong:
            external = builder.CreateVector(std::vector<int64_t>{0, externalLength});
            break;
        case BlobKind::HalfExternalTooLong:
            type = DataType_DT_HALF;
            bytes = builder.CreateVector(std::vector<uint8_t>{0});
            external = builder.CreateVector(std::vector<int64_t>{0, externalLength});
            break;
        case BlobKind::HalfInlineShort:
            type = DataType_DT_HALF;
            bytes = builder.CreateVector(std::vector<uint8_t>{0, 0});
            break;
        case BlobKind::Valid:
            floats = builder.CreateVector(std::vector<float>{1.f, 2.f, 3.f, 4.f});
            break;
    }
    BlobBuilder blob(builder);
    blob.add_dims(dims);
    blob.add_dataType(type);
    if (!floats.IsNull())
        blob.add_float32s(floats);
    if (!bytes.IsNull())
        blob.add_uint8s(bytes);
    if (!external.IsNull())
        blob.add_external(external);
    const auto payload = blob.Finish();
    const auto output = builder.CreateVector(std::vector<int>{0});
    OpBuilder op(builder);
    op.add_type(OpType_Const);
    op.add_main_type(OpParameter_Blob);
    op.add_main(payload.Union());
    op.add_outputIndexes(output);
    const auto constOp = op.Finish();
    const auto ops = builder.CreateVector(std::vector<flatbuffers::Offset<Op>>{constOp});
    const auto names =
        builder.CreateVector(std::vector<flatbuffers::Offset<flatbuffers::String>>{builder.CreateString("out")});
    NetBuilder net(builder);
    net.add_oplists(ops);
    net.add_tensorName(names);
    net.add_outputName(names);
    net.add_tensorNumber(1);
    builder.Finish(net.Finish());
    return std::vector<uint8_t>(builder.GetBufferPointer(), builder.GetBufferPointer() + builder.GetSize());
}

static bool loads(const std::vector<uint8_t>& graph, std::shared_ptr<Executor::RuntimeManager> runtime) {
    std::unique_ptr<Module> module(Module::load({}, {"out"}, graph.data(), graph.size(), runtime));
    if (!module) {
        return false;
    }
    auto out = module->onForward({});
    return out.size() == 1 && nullptr != out[0]->readMap<float>();
}

class ConstBlobBoundsTest : public MNNTestCase {
public:
    bool run(int precision) override {
        char path[] = MNN_BLOB_BOUNDS_TMP "mnn-blob-bounds-XXXXXX";
        const int fd = mkstemp(path);
        if (fd < 0) {
            return false;
        }
        const int64_t big = 64 * 1024;
        std::vector<char> zeros(big, 0);
        const bool written = write(fd, zeros.data(), zeros.size()) == (ssize_t)zeros.size();
        close(fd);
        ScheduleConfig config;
        config.numThread = 1;
        std::shared_ptr<Executor::RuntimeManager> runtime(Executor::RuntimeManager::createRuntimeManager(config));
        bool ok = written && runtime != nullptr;
        if (ok) {
            runtime->setExternalFile(path);
            ok = loads(constGraph(BlobKind::Valid, 0), runtime);
            MNN_PRINT("valid inline blob loads: %d\n", ok);
            for (auto kind : {BlobKind::InlineShort, BlobKind::InlineMissing, BlobKind::ExternalTooLong,
                              BlobKind::HalfExternalTooLong, BlobKind::HalfInlineShort}) {
                const bool loaded = loads(constGraph(kind, big), runtime);
                MNN_PRINT("malformed blob kind %d rejected: %d\n", (int)kind, !loaded);
                ok = ok && !loaded;
            }
            // Exact-size external data still loads (4 floats = 16 bytes).
            const bool exact = loads(constGraph(BlobKind::ExternalTooLong, 16), runtime);
            MNN_PRINT("exact-size external blob loads: %d\n", exact);
            ok = ok && exact;
        }
        unlink(path);
        return ok;
    }
};
MNNTestSuiteRegister(ConstBlobBoundsTest, "core/const_blob_bounds");
#endif
