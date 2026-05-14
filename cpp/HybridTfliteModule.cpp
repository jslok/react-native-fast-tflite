#include "HybridTfliteModule.hpp"
#include "TfliteHelpers.hpp"
#include "TfliteWorker.hpp"

#include <functional>

#if defined(ANDROID)
#include <tflite/c/c_api.h>
#include <tflite/delegates/gpu/delegate.h>
#include <tflite/delegates/nnapi/nnapi_delegate_c_api.h>
#elif defined(__APPLE__)
#include <TensorFlowLiteC/TensorFlowLiteC.h>
#if FAST_TFLITE_ENABLE_CORE_ML
#include <TensorFlowLiteCCoreML/TensorFlowLiteCCoreML.h>
#endif
#else
#error "Invalid Platform!"
#endif

namespace margelo::nitro::tflite {

/**
 * Return a Hardware accelerated delegate, or throws
 * if the given delegate type is not available.
 */
TfLiteDelegate* getDelegate(TensorflowModelDelegate delegateType) {
  switch (delegateType) {
    case TensorflowModelDelegate::CORE_ML:
      return getCoreMLDelegate();
    case TensorflowModelDelegate::METAL:
      return getMetalDelegate();
    case TensorflowModelDelegate::NNAPI:
      return getNNAPIDelegate();
    case TensorflowModelDelegate::ANDROID_GPU:
      return getAndroidGPUDelegate();
  }
  throw std::runtime_error("Unknown Delegate \"" + std::to_string(static_cast<int>(delegateType)) +
                           "\"!");
}

/**
 * Pair the given delegate pointer with the matching TFLite C API destroy
 * function so the model destructor can run it later on the worker thread.
 * Returns `nullptr` if the platform / build doesn't provide a deleter, in
 * which case the delegate construction above would already have failed.
 */
static std::function<void()> makeDelegateDeleter(TensorflowModelDelegate delegateType,
                                                 TfLiteDelegate* delegate) {
  switch (delegateType) {
    case TensorflowModelDelegate::CORE_ML:
#if defined(__APPLE__) && FAST_TFLITE_ENABLE_CORE_ML
      return [delegate]() { TfLiteCoreMlDelegateDelete(delegate); };
#else
      break;
#endif
    case TensorflowModelDelegate::METAL:
      // getMetalDelegate() currently throws, so this case is unreachable in
      // practice — but be defensive if a future patch wires it up.
      break;
    case TensorflowModelDelegate::NNAPI:
#if defined(ANDROID)
      return [delegate]() { TfLiteNnapiDelegateDelete(delegate); };
#else
      break;
#endif
    case TensorflowModelDelegate::ANDROID_GPU:
#if defined(ANDROID)
      return [delegate]() { TfLiteGpuDelegateV2Delete(delegate); };
#else
      break;
#endif
  }
  return nullptr;
}

std::shared_ptr<HybridTfliteModelSpec>
HybridTfliteModule::createModel(const std::shared_ptr<ArrayBuffer>& modelData,
                                const std::vector<TensorflowModelDelegate>& delegates) {
  // Confine the whole model + delegate + interpreter creation pipeline (and
  // the initial AllocateTensors) to the dedicated worker thread. This binds
  // any GPU delegate's CL context to that single thread, so the matching
  // destruction in ~HybridTfliteModel — also routed through the worker —
  // runs on the same OS thread. Required by PowerVR drivers; see
  // https://github.com/mrousavy/react-native-fast-tflite/issues/186 .
  return worker::run([&modelData, &delegates]() -> std::shared_ptr<HybridTfliteModelSpec> {
    TfLiteModel* model = TfLiteModelCreate(modelData->data(), modelData->size());
    if (model == nullptr) {
      throw std::runtime_error("Failed to create TFLite model from data!");
    }

    // Configure interpreter via options.
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

    // Add each hardware-accelerated delegate AND capture its matching destroy
    // function so HybridTfliteModel can run them later on this same worker
    // thread. The default CPU delegate needs no explicit cleanup.
    std::vector<std::function<void()>> delegateDeleters;
    delegateDeleters.reserve(delegates.size());
    for (const TensorflowModelDelegate& delegateType : delegates) {
      TfLiteDelegate* delegate = getDelegate(delegateType);
      TfLiteInterpreterOptionsAddDelegate(options, delegate);
      auto deleter = makeDelegateDeleter(delegateType, delegate);
      if (deleter) delegateDeleters.push_back(std::move(deleter));
    }

    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    // Options and model object can be deleted immediately after interpreter creation.
    // (per TFLite C API docs — the model_data buffer must still outlive the interpreter,
    // which is handled by _modelData shared_ptr in HybridTfliteModel)
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    if (interpreter == nullptr) {
      // Tear down each delegate we just created — still on the worker.
      for (auto it = delegateDeleters.rbegin(); it != delegateDeleters.rend(); ++it) {
        if (*it) (*it)();
      }
      throw std::runtime_error("Failed to create TFLite interpreter!");
    }

    // Allocate tensors on the worker thread — this is where GPU buffers are
    // initialized and bound to the delegate's CL context (PVR-sensitive).
    TfLiteStatus status = TfLiteInterpreterAllocateTensors(interpreter);
    if (status != kTfLiteOk) {
      TfLiteInterpreterDelete(interpreter);
      for (auto it = delegateDeleters.rbegin(); it != delegateDeleters.rend(); ++it) {
        if (*it) (*it)();
      }
      throw std::runtime_error(
          "TFLite: Failed to allocate memory for input/output tensors! Status: " +
          tfLiteStatusToString(status));
    }

    // Wrap in HybridTfliteModel — stores shared_ptr<ArrayBuffer> to keep model data
    // bytes alive, and the captured deleters for thread-affined teardown.
    return std::make_shared<HybridTfliteModel>(
        interpreter, modelData, delegates, std::move(delegateDeleters));
  });
}

} // namespace margelo::nitro::tflite
