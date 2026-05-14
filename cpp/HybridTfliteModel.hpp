#pragma once

#include "HybridTfliteModelSpec.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(ANDROID)
#include <tflite/c/c_api.h>
#elif defined(__APPLE__)
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#else
#error "Invalid Platform!"
#endif

namespace margelo::nitro::tflite {

class HybridTfliteModel : public HybridTfliteModelSpec {
public:
  /**
   * Construct with a pre-built interpreter and the deleters needed to tear
   * down each delegate that was attached to it. Caller (HybridTfliteModule)
   * is responsible for running both construction and the
   * `TfLiteInterpreterAllocateTensors` call on the dedicated worker thread so
   * that the GPU delegate's CL context is bound to that thread.
   */
  explicit HybridTfliteModel(TfLiteInterpreter* interpreter, std::shared_ptr<ArrayBuffer> modelData,
                             std::vector<TensorflowModelDelegate> delegates,
                             std::vector<std::function<void()>> delegateDeleters);
  ~HybridTfliteModel();

  // Properties (from HybridTfliteModelSpec)
  std::vector<TensorflowModelDelegate> getDelegates() override;
  std::vector<Tensor> getInputs() override;
  std::vector<Tensor> getOutputs() override;

  // Methods (from HybridTfliteModelSpec)
  std::vector<std::shared_ptr<ArrayBuffer>>
  runSync(const std::vector<std::shared_ptr<ArrayBuffer>>& input) override;
  std::shared_ptr<Promise<std::vector<std::shared_ptr<ArrayBuffer>>>>
  run(const std::vector<std::shared_ptr<ArrayBuffer>>& input) override;

private:
  void copyInputBuffers(const std::vector<std::shared_ptr<ArrayBuffer>>& input);
  void invoke();
  std::vector<std::shared_ptr<ArrayBuffer>> copyOutputBuffers();
  std::shared_ptr<ArrayBuffer> getOutputBufferForTensor(const TfLiteTensor* tensor);

private:
  TfLiteInterpreter* _interpreter = nullptr;
  std::vector<TensorflowModelDelegate> _delegates;
  std::shared_ptr<ArrayBuffer> _modelData;
  std::unordered_map<std::string, std::shared_ptr<ArrayBuffer>> _outputBuffers;
  // Deleters for each delegate attached to `_interpreter`, in creation order.
  // Run in reverse during destruction (on the worker thread) so each delegate
  // is destroyed on the same OS thread that created it.
  std::vector<std::function<void()>> _delegateDeleters;
};

} // namespace margelo::nitro::tflite
