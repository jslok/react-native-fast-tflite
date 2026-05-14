#pragma once

#include <functional>
#include <future>
#include <memory>
#include <type_traits>
#include <utility>

namespace margelo::nitro::tflite::worker {

/**
 * @return `true` if the caller is currently executing on the dedicated
 * TFLite worker thread. Used by `run()` to detect re-entry and avoid
 * self-deadlock.
 */
bool isOnWorkerThread();

namespace detail {
/**
 * Post a task to the worker thread queue and return immediately.
 * The worker thread is created lazily on first call and runs for the
 * lifetime of the process.
 */
void post(std::function<void()> task);
} // namespace detail

/**
 * Run `task` on the dedicated TFLite worker thread and BLOCK the caller
 * until it completes. Returns the value produced by `task`, or rethrows
 * if `task` threw.
 *
 * All TFLite C API calls (delegate create/destroy, interpreter
 * create/invoke/destroy, tensor allocate) must be routed through this
 * thread on Android to satisfy PowerVR's OpenCL driver invariant that
 * the CL context's create / use / destroy happen on the same OS thread.
 * Mixing threads causes a SIGABRT in the PVR CDM watchdog. See:
 *   https://github.com/mrousavy/react-native-fast-tflite/issues/186
 *
 * If called from within a task that is already running on the worker
 * thread, `run()` executes inline to avoid deadlocking on itself.
 */
template <typename F>
auto run(F&& task) -> decltype(task()) {
  using Result = decltype(task());

  // Re-entry: already on the worker thread, run inline.
  if (isOnWorkerThread()) {
    return std::forward<F>(task)();
  }

  // packaged_task wraps the callable so we can extract a future. It is
  // move-only, so we hold it in a shared_ptr to satisfy std::function's
  // CopyConstructible requirement when the lambda below is type-erased.
  auto packaged = std::make_shared<std::packaged_task<Result()>>(std::forward<F>(task));
  std::future<Result> future = packaged->get_future();

  detail::post([packaged]() { (*packaged)(); });

  return future.get(); // blocks; rethrows on exception from task
}

} // namespace margelo::nitro::tflite::worker
