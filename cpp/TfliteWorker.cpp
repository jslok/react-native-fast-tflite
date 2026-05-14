#include "TfliteWorker.hpp"

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

#if defined(ANDROID)
#include <pthread.h>
#endif

namespace margelo::nitro::tflite::worker {

namespace {

struct WorkerState {
  std::mutex mutex;
  std::condition_variable cv;
  std::queue<std::function<void()>> queue;
  bool started = false;
};

// Intentionally leaked: the worker thread lives for the entire process and
// must observe a valid mutex / condition_variable / queue even after static
// destructors would otherwise run. Allocating on the heap and never freeing
// guarantees that.
WorkerState* getState() {
  static WorkerState* state = new WorkerState();
  return state;
}

thread_local bool t_isWorkerThread = false;

void workerLoop(WorkerState* state) {
  t_isWorkerThread = true;

#if defined(ANDROID)
  pthread_setname_np(pthread_self(), "TfliteWorker");
#endif

  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(state->mutex);
      state->cv.wait(lock, [state]() { return !state->queue.empty(); });
      task = std::move(state->queue.front());
      state->queue.pop();
    }
    // Each posted task is a std::packaged_task wrapper that captures its own
    // exception into a std::future, so we never propagate exceptions to the
    // worker loop itself.
    task();
  }
}

} // namespace

bool isOnWorkerThread() {
  return t_isWorkerThread;
}

namespace detail {

void post(std::function<void()> task) {
  WorkerState* state = getState();
  bool needsStart = false;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (!state->started) {
      state->started = true;
      needsStart = true;
    }
    state->queue.push(std::move(task));
  }
  if (needsStart) {
    // Detached: the worker thread is a singleton with no clean shutdown path.
    // Tied to the leaked WorkerState, so synchronization primitives stay valid.
    std::thread(workerLoop, state).detach();
  }
  state->cv.notify_one();
}

} // namespace detail

} // namespace margelo::nitro::tflite::worker
