#include "HybridTfliteModel.hpp"
#include "TfliteHelpers.hpp"
#include "TfliteWorker.hpp"

#include <NitroModules/ArrayBuffer.hpp>
#include <NitroModules/Promise.hpp>
#include <string>

#if defined(ANDROID)
#include <tflite/c/c_api.h>
#elif defined(__APPLE__)
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#else
#error "Invalid Platform!"
#endif

namespace margelo::nitro::tflite {

HybridTfliteModel::HybridTfliteModel(TfLiteInterpreter* interpreter,
                                     std::shared_ptr<ArrayBuffer> modelData,
                                     std::vector<TensorflowModelDelegate> delegates,
                                     std::vector<std::function<void()>> delegateDeleters)
    : HybridObject(TAG), _interpreter(interpreter), _delegates(std::move(delegates)),
      _modelData(modelData), _delegateDeleters(std::move(delegateDeleters)) {
  // The caller (HybridTfliteModule::createModel) is responsible for calling
  // `TfLiteInterpreterAllocateTensors` on the worker thread before constructing
  // us. AllocateTensors initializes per-delegate GPU buffers and must run on
  // the same OS thread that created the delegate (see TfliteWorker.hpp).
}

HybridTfliteModel::~HybridTfliteModel() {
  // The destructor runs on whichever thread Hermes' JSI finalizer reaps this
  // object from — usually NOT the thread that created the GPU delegate. Route
  // teardown through the worker so create / destroy run on the same OS thread,
  // satisfying PowerVR's per-thread OpenCL context invariant.
  TfLiteInterpreter* interpreter = _interpreter;
  std::vector<std::function<void()>> deleters = std::move(_delegateDeleters);
  _interpreter = nullptr;
  if (interpreter == nullptr && deleters.empty()) return;

  try {
    worker::run([interpreter, deleters = std::move(deleters)]() {
      if (interpreter != nullptr) {
        TfLiteInterpreterDelete(interpreter);
      }
      // Delete delegates in reverse creation order, after the interpreter.
      // Each deleter calls the destroy function paired with the delegate at
      // construction (TfLiteGpuDelegateV2Delete, TfLiteNnapiDelegateDelete, ...)
      for (auto it = deleters.rbegin(); it != deleters.rend(); ++it) {
        if (*it) (*it)();
      }
    });
  } catch (...) {
    // Destructors must not throw.
  }
  // _modelData (shared_ptr<ArrayBuffer>) is automatically freed
}

std::vector<TensorflowModelDelegate> HybridTfliteModel::getDelegates() {
  return _delegates;
}

std::vector<Tensor> HybridTfliteModel::getInputs() {
  int count = TfLiteInterpreterGetInputTensorCount(_interpreter);
  std::vector<Tensor> tensors;
  tensors.reserve(count);
  for (int32_t i = 0; i < count; i++) {
    TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(_interpreter, i);
    if (tensor == nullptr) {
      throw std::runtime_error("TFLite: Failed to get input tensor " + std::to_string(i) + "!");
    }
    int dimensions = TfLiteTensorNumDims(tensor);
    std::vector<double> shape;
    shape.reserve(dimensions);
    for (int32_t d = 0; d < dimensions; d++) {
      int32_t size = TfLiteTensorDim(tensor, d);
      shape.push_back(static_cast<double>(size));
    }
    tensors.push_back(Tensor(std::string(TfLiteTensorName(tensor)),
                             getTensorDataType(TfLiteTensorType(tensor)), std::move(shape)));
  }
  return tensors;
}

std::vector<Tensor> HybridTfliteModel::getOutputs() {
  int count = TfLiteInterpreterGetOutputTensorCount(_interpreter);
  std::vector<Tensor> tensors;
  tensors.reserve(count);
  for (int32_t i = 0; i < count; i++) {
    const TfLiteTensor* tensor = TfLiteInterpreterGetOutputTensor(_interpreter, i);
    if (tensor == nullptr) {
      throw std::runtime_error("TFLite: Failed to get output tensor " + std::to_string(i) + "!");
    }
    int dimensions = TfLiteTensorNumDims(tensor);
    std::vector<double> shape;
    shape.reserve(dimensions);
    for (int32_t d = 0; d < dimensions; d++) {
      int32_t size = TfLiteTensorDim(tensor, d);
      shape.push_back(static_cast<double>(size));
    }
    tensors.push_back(Tensor(std::string(TfLiteTensorName(tensor)),
                             getTensorDataType(TfLiteTensorType(tensor)), std::move(shape)));
  }
  return tensors;
}

void HybridTfliteModel::copyInputBuffers(const std::vector<std::shared_ptr<ArrayBuffer>>& input) {
  size_t inputCount = TfLiteInterpreterGetInputTensorCount(_interpreter);
  if (input.size() != inputCount) [[unlikely]] {
    throw std::runtime_error("TFLite: Input array size (" + std::to_string(input.size()) +
                             ") does not match input tensor count (" + std::to_string(inputCount) +
                             ")!");
  }

  for (int32_t i = 0; i < inputCount; i++) {
    TfLiteTensor* tensor = TfLiteInterpreterGetInputTensor(_interpreter, i);
    const std::shared_ptr<ArrayBuffer>& buffer = input[i];
    TfLiteTensorCopyFromBuffer(tensor, buffer->data(), buffer->size());
  }
}

// Pre-allocates a buffer per output tensor, reuses on subsequent calls.
std::shared_ptr<ArrayBuffer>
HybridTfliteModel::getOutputBufferForTensor(const TfLiteTensor* tensor) {
  std::string name = TfLiteTensorName(tensor);
  if (_outputBuffers.find(name) == _outputBuffers.end()) {
    TfLiteType dataType = TfLiteTensorType(tensor);
    int totalLength = getTensorTotalLength(tensor);
    size_t byteSize = totalLength * getTFLTensorDataTypeSize(dataType);
    _outputBuffers[name] = ArrayBuffer::allocate(byteSize);
  }
  return _outputBuffers[name];
}

std::vector<std::shared_ptr<ArrayBuffer>> HybridTfliteModel::copyOutputBuffers() {
  int outputCount = TfLiteInterpreterGetOutputTensorCount(_interpreter);
  std::vector<std::shared_ptr<ArrayBuffer>> results;
  results.reserve(outputCount);

  for (int32_t i = 0; i < outputCount; i++) {
    const TfLiteTensor* tensor = TfLiteInterpreterGetOutputTensor(_interpreter, i);
    std::shared_ptr<ArrayBuffer> outputBuffer = getOutputBufferForTensor(tensor);

    std::string name = TfLiteTensorName(tensor);

    void* tensorData = TfLiteTensorData(tensor);
    if (tensorData == nullptr) {
      [[unlikely]];
      throw std::runtime_error("TFLite: Failed to get data from tensor \"" + name + "\"!");
    }

    memcpy(outputBuffer->data(), tensorData, outputBuffer->size());
    results.push_back(outputBuffer);
  }

  return results;
}

void HybridTfliteModel::invoke() {
  TfLiteStatus status = TfLiteInterpreterInvoke(_interpreter);
  if (status != kTfLiteOk) {
    throw std::runtime_error("TFLite: Failed to run TFLite Model! Status: " +
                             tfLiteStatusToString(status));
  }
}

std::vector<std::shared_ptr<ArrayBuffer>>
HybridTfliteModel::runSync(const std::vector<std::shared_ptr<ArrayBuffer>>& input) {
  // Confine the full input-copy → invoke → output-copy pipeline to the worker
  // thread so all interpreter touches happen on the same OS thread that created
  // the GPU delegate. worker::run blocks the caller, so the JS-owned input
  // ArrayBuffers stay alive throughout (no GC race).
  return worker::run([this, &input]() -> std::vector<std::shared_ptr<ArrayBuffer>> {
    copyInputBuffers(input);
    invoke();
    return copyOutputBuffers();
  });
}

std::shared_ptr<Promise<std::vector<std::shared_ptr<ArrayBuffer>>>>
HybridTfliteModel::run(const std::vector<std::shared_ptr<ArrayBuffer>>& input) {
  // Copy input on the worker before returning — JS-owned ArrayBuffers cannot
  // safely outlive this call because the Promise's async continuation runs
  // on a different thread and the input may be GC'd. worker::run blocks
  // briefly while the (cheap, memcpy-based) copy completes.
  worker::run([this, &input]() { copyInputBuffers(input); });

  std::shared_ptr<HybridTfliteModel> sharedThis = shared_cast<HybridTfliteModel>();
  return Promise<std::vector<std::shared_ptr<ArrayBuffer>>>::async(
      [sharedThis]() -> std::vector<std::shared_ptr<ArrayBuffer>> {
        return worker::run([sharedThis]() -> std::vector<std::shared_ptr<ArrayBuffer>> {
          sharedThis->invoke();
          return sharedThis->copyOutputBuffers();
        });
      });
}

} // namespace margelo::nitro::tflite
