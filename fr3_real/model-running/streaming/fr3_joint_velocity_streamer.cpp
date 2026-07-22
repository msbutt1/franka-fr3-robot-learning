// Persistent libfranka joint-velocity streamer for timestamped OpenPI plans.
//
// Policy plans carry the monotonic timestamp of the robot state used for
// inference. A newly validated plan replaces the current plan immediately at
// the action index implied by its age. The 1 kHz FCI thread performs no file or
// console I/O; fixed-size telemetry events are printed by a separate thread.

#include <arpa/inet.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

#include <franka/active_control_base.h>
#include <franka/control_types.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>

namespace {
constexpr uint32_t kCommandMagic = 0x46523343;  // FR3C
constexpr uint32_t kStateMagic = 0x46523353;    // FR3S
constexpr uint32_t kVersion = 3;
constexpr size_t kJoints = 7;
constexpr size_t kMaxSteps = 16;
constexpr size_t kEventCapacity = 128;
#ifdef FR3_REFERENCE_CONTROLLER
constexpr bool kReferenceController = true;
constexpr const char* kControllerName = "libfranka_reference_rate_limiter";
#else
constexpr bool kReferenceController = false;
constexpr const char* kControllerName = "custom_velocity_tracker";
#endif

enum class StopReason : uint32_t {
  kNone = 0,
  kCommand = 1,
  kSignal = 2,
  kJointGuard = 3,
  kWorkspaceGuard = 4,
  kWatchdog = 5,
  kPredictedGuard = 6,
  kStalePlan = 7,
};

const char* stop_reason_name(StopReason reason) {
  switch (reason) {
    case StopReason::kNone: return "none";
    case StopReason::kCommand: return "command";
    case StopReason::kSignal: return "signal";
    case StopReason::kJointGuard: return "joint_guard";
    case StopReason::kWorkspaceGuard: return "workspace_guard";
    case StopReason::kWatchdog: return "watchdog";
    case StopReason::kPredictedGuard: return "predicted_guard";
    case StopReason::kStalePlan: return "stale_plan";
  }
  return "unknown";
}

#pragma pack(push, 1)
struct CommandPacket {
  uint32_t magic;
  uint32_t version;
  uint64_t sequence;
  uint64_t observation_monotonic_ns;
  uint8_t stop;
  uint8_t steps;
  uint16_t reserved;
  double velocities[kMaxSteps][kJoints];
};

struct StatePacket {
  uint32_t magic;
  uint32_t version;
  uint64_t monotonic_ns;
  double q[kJoints];
  double dq[kJoints];
  double tcp_xyz[3];
  uint64_t active_sequence;
  uint32_t plan_step;
  uint32_t plan_steps;
  uint32_t activation_offset;
  uint32_t stop_reason;
};
#pragma pack(pop)

static_assert(sizeof(CommandPacket) == 924, "Unexpected command packet layout");
static_assert(sizeof(StatePacket) == 176, "Unexpected state packet layout");

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true, std::memory_order_relaxed); }

uint64_t monotonic_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count());
}

size_t available_cpu_count() {
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  if (sched_getaffinity(0, sizeof(cpu_set), &cpu_set) != 0) return 0;
  return static_cast<size_t>(CPU_COUNT(&cpu_set));
}

double clamp(double value, double lower, double upper) {
  return std::max(lower, std::min(value, upper));
}

struct Limits {
  std::array<double, kJoints> velocity{};
  std::array<double, kJoints> acceleration{};
  std::array<double, kJoints> jerk{};
  std::array<double, kJoints> q_min{{-2.6237, -1.6637, -2.7807, -2.9221, -2.6865, 0.6645, -2.8959}};
  std::array<double, kJoints> q_max{{2.6237, 1.6637, 2.7807, -0.2718, 2.6865, 4.3969, 2.8959}};
};

struct Workspace {
  std::array<double, 3> minimum{{0.18, -0.50, 0.03}};
  std::array<double, 3> maximum{{0.75, 0.45, 0.65}};
  std::array<double, 3> guard_margin{{0.005, 0.005, 0.002}};
  // Reserve enough space to decelerate before a hard workspace boundary.
  double braking_acceleration = 1.0;
  double reaction_time_s = 0.020;
  double braking_buffer = 0.002;
};

double braking_distance(double speed_toward_boundary, const Workspace& workspace) {
  const double speed = std::max(0.0, speed_toward_boundary);
  return workspace.braking_buffer + speed * workspace.reaction_time_s +
         speed * speed / (2.0 * workspace.braking_acceleration);
}

struct Mailbox {
  std::array<std::array<std::array<std::atomic<double>, kJoints>, kMaxSteps>, 2> velocity{};
  std::array<std::atomic<uint8_t>, 2> steps{{0, 0}};
  std::array<std::atomic<uint64_t>, 2> sequence{{0, 0}};
  std::array<std::atomic<uint64_t>, 2> observation_monotonic_ns{{0, 0}};
  std::atomic<int> active_buffer{0};
  std::atomic<uint64_t> generation{0};
  std::atomic<bool> stop_requested{false};
};

struct RobotSnapshot {
  std::atomic<uint64_t> sequence{0};
  std::array<std::atomic<double>, kJoints> q{};
  std::array<std::atomic<double>, kJoints> dq_d{};
  std::array<std::atomic<double>, kJoints> ddq_d{};
  std::array<std::atomic<double>, 16> F_T_EE{};
  std::array<std::atomic<double>, 16> EE_T_K{};
};

struct StateOutputMailbox {
  std::atomic<uint64_t> sequence{0};
  std::atomic<uint64_t> monotonic_ns{0};
  std::array<std::atomic<double>, kJoints> q{};
  std::array<std::atomic<double>, kJoints> dq{};
  std::array<std::atomic<double>, 3> tcp_xyz{};
  std::atomic<uint64_t> active_sequence{0};
  std::atomic<uint32_t> plan_step{0};
  std::atomic<uint32_t> plan_steps{0};
  std::atomic<uint32_t> activation_offset{0};
  std::atomic<uint32_t> stop_reason{0};
};

struct GuardState {
  std::array<double, kJoints> q{};
  std::array<double, kJoints> dq_d{};
  std::array<double, kJoints> ddq_d{};
  std::array<double, 16> F_T_EE{};
  std::array<double, 16> EE_T_K{};
};

struct ValidatedMailbox {
  std::array<std::array<std::array<std::atomic<double>, kJoints>, kMaxSteps>, 2> velocity{};
  std::array<std::array<std::array<std::atomic<double>, 3>, kMaxSteps>, 2> predicted_tcp{};
  std::array<std::atomic<uint8_t>, 2> steps{{0, 0}};
  std::array<std::atomic<uint8_t>, 2> checked_offset{{0, 0}};
  std::array<std::atomic<uint8_t>, 2> predicted_count{{0, 0}};
  std::array<std::atomic<uint8_t>, 2> rejected_step{{0, 0}};
  std::array<std::atomic<uint64_t>, 2> sequence{{0, 0}};
  std::array<std::atomic<uint64_t>, 2> observation_monotonic_ns{{0, 0}};
  std::array<std::atomic<uint64_t>, 2> source_generation{{0, 0}};
  std::array<std::atomic<uint32_t>, 2> reason{{0, 0}};
  std::array<std::atomic<bool>, 2> safe{{false, false}};
  std::atomic<int> active_buffer{0};
  std::atomic<uint64_t> generation{0};
};

enum class EventType : uint8_t { kPlanAccepted, kPlanRejected, kStop };

struct TelemetryEvent {
  EventType type = EventType::kPlanAccepted;
  StopReason reason = StopReason::kNone;
  uint64_t sequence = 0;
  uint8_t steps = 0;
  uint8_t activation_offset = 0;
  double observation_age_ms = 0.0;
  double raw_peak = 0.0;
  std::array<double, kJoints> raw_joint_peak{};
  std::array<uint32_t, kJoints> clipping_count{};
  std::array<double, 3> actual_tcp{};
  std::array<std::array<double, 3>, kMaxSteps> predicted_tcp{};
  uint8_t predicted_count = 0;
  uint8_t rejected_step = 0;
};

struct EventQueue {
  std::array<TelemetryEvent, kEventCapacity> events{};
  std::atomic<uint64_t> write_index{0};
  std::atomic<uint64_t> read_index{0};

  void push(const TelemetryEvent& event) {
    const uint64_t write = write_index.load(std::memory_order_relaxed);
    const uint64_t read = read_index.load(std::memory_order_acquire);
    if (write - read >= kEventCapacity) return;
    events[write % kEventCapacity] = event;
    write_index.store(write + 1, std::memory_order_release);
  }

  bool pop(TelemetryEvent* event) {
    const uint64_t read = read_index.load(std::memory_order_relaxed);
    if (read == write_index.load(std::memory_order_acquire)) return false;
    *event = events[read % kEventCapacity];
    read_index.store(read + 1, std::memory_order_release);
    return true;
  }
};

void print_event(const TelemetryEvent& event) {
  if (event.type == EventType::kStop) {
    std::cout << "[STREAM] stop reason=" << stop_reason_name(event.reason)
              << " actual_tcp=[" << event.actual_tcp[0] << "," << event.actual_tcp[1]
              << "," << event.actual_tcp[2] << "]" << std::endl;
    return;
  }
  std::cout << "[STREAM] plan seq=" << event.sequence
            << (event.type == EventType::kPlanAccepted ? " accepted" : " rejected")
            << " steps=" << int(event.steps)
            << " observation_age_ms=" << event.observation_age_ms
            << " activation_offset=" << int(event.activation_offset)
            << " raw_peak=" << event.raw_peak << " clips=[";
  for (size_t joint = 0; joint < kJoints; ++joint) {
    std::cout << event.clipping_count[joint] << (joint + 1 == kJoints ? "]" : ",");
  }
  std::cout << " raw_joint_peak=[";
  for (size_t joint = 0; joint < kJoints; ++joint) {
    std::cout << event.raw_joint_peak[joint] << (joint + 1 == kJoints ? "]" : ",");
  }
  std::cout << " actual_tcp=[" << event.actual_tcp[0] << "," << event.actual_tcp[1]
            << "," << event.actual_tcp[2] << "] predicted_tcp=[";
  for (size_t step = 0; step < event.predicted_count; ++step) {
    const auto& xyz = event.predicted_tcp[step];
    std::cout << "[" << xyz[0] << "," << xyz[1] << "," << xyz[2] << "]"
              << (step + 1 == event.predicted_count ? "" : ",");
  }
  std::cout << "]";
  if (event.type == EventType::kPlanRejected) {
    std::cout << " reason=" << stop_reason_name(event.reason)
              << " rejected_action=" << int(event.rejected_step);
  }
  std::cout << std::endl;
}

void event_logger_loop(EventQueue* queue, std::atomic<bool>* done) {
  while (!done->load(std::memory_order_acquire) ||
         queue->read_index.load(std::memory_order_acquire) !=
             queue->write_index.load(std::memory_order_acquire)) {
    TelemetryEvent event{};
    if (queue->pop(&event)) {
      print_event(event);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  }
}

int make_bound_socket(uint16_t port) {
  const int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) throw std::runtime_error("Could not create UDP socket");
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    close(fd);
    throw std::runtime_error("Could not bind loopback UDP command socket");
  }
  timeval receive_timeout{};
  receive_timeout.tv_usec = 100000;
  if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &receive_timeout, sizeof(receive_timeout)) != 0) {
    close(fd);
    throw std::runtime_error("Could not set UDP command socket timeout");
  }
  return fd;
}

void receiver_loop(int socket_fd, Mailbox* mailbox) {
  while (!g_stop.load(std::memory_order_relaxed)) {
    CommandPacket packet{};
    const ssize_t received = recv(socket_fd, &packet, sizeof(packet), 0);
    if (received < 0) continue;
    if (static_cast<size_t>(received) != sizeof(packet) || packet.magic != kCommandMagic ||
        packet.version != kVersion) {
      continue;
    }
    if (packet.stop != 0) {
      mailbox->stop_requested.store(true, std::memory_order_release);
      continue;
    }
    if (packet.steps == 0 || packet.steps > kMaxSteps || packet.observation_monotonic_ns == 0) continue;
    const int write_buffer = 1 - mailbox->active_buffer.load(std::memory_order_relaxed);
    for (size_t step = 0; step < kMaxSteps; ++step) {
      for (size_t joint = 0; joint < kJoints; ++joint) {
        const double value = step < packet.steps ? packet.velocities[step][joint] : 0.0;
        mailbox->velocity[write_buffer][step][joint].store(value, std::memory_order_relaxed);
      }
    }
    mailbox->steps[write_buffer].store(packet.steps, std::memory_order_relaxed);
    mailbox->sequence[write_buffer].store(packet.sequence, std::memory_order_relaxed);
    mailbox->observation_monotonic_ns[write_buffer].store(
        packet.observation_monotonic_ns, std::memory_order_relaxed);
    mailbox->active_buffer.store(write_buffer, std::memory_order_release);
    mailbox->generation.fetch_add(1, std::memory_order_release);
  }
}

struct ThreadCleanup {
  int command_socket;
  int state_socket;
  std::thread& receiver;
  std::thread& logger;
  std::atomic<bool>& logger_done;

  ~ThreadCleanup() {
    g_stop.store(true, std::memory_order_relaxed);
    if (command_socket >= 0) close(command_socket);
    if (state_socket >= 0) close(state_socket);
    if (receiver.joinable()) receiver.join();
    logger_done.store(true, std::memory_order_release);
    if (logger.joinable()) logger.join();
  }
};

struct ValidatorCleanup {
  std::atomic<bool>& done;
  std::thread& validator;

  ~ValidatorCleanup() {
    done.store(true, std::memory_order_release);
    if (validator.joinable()) validator.join();
  }
};

struct StatePublisherCleanup {
  std::atomic<bool>& done;
  std::thread& publisher;

  ~StatePublisherCleanup() {
    done.store(true, std::memory_order_release);
    if (publisher.joinable()) publisher.join();
  }
};

struct Args {
  std::string robot_ip;
  uint16_t command_port = 51000;
  uint16_t state_port = 51001;
  double policy_hz = 15.0;
  double state_hz = 60.0;
  double watchdog_s = 0.75;
  double stop_settle_s = 0.30;
  double velocity_tracking_gain = 20.0;
  Limits limits;
  Workspace workspace;
  bool dry_run = false;
};

Args parse_args(int argc, char** argv) {
  if (argc < 2) throw std::runtime_error("Usage: fr3_joint_velocity_streamer <robot-ip> [options]");
  Args args;
  args.robot_ip = argv[1];
  args.limits.velocity.fill(0.10);
  args.limits.acceleration.fill(0.75);
  if constexpr (kReferenceController) {
    args.limits.jerk = franka::kMaxJointJerk;
  } else {
    args.limits.jerk.fill(30.0);
  }
  for (int index = 2; index < argc; ++index) {
    const std::string key = argv[index];
    if (key == "--dry-run") {
      args.dry_run = true;
      continue;
    }
    auto value = [&]() -> std::string {
      if (++index >= argc) throw std::runtime_error("Missing value for " + key);
      return argv[index];
    };
    if (key == "--command-port") {
      args.command_port = static_cast<uint16_t>(std::stoi(value()));
    } else if (key == "--state-port") {
      args.state_port = static_cast<uint16_t>(std::stoi(value()));
    } else if (key == "--policy-hz") {
      args.policy_hz = std::stod(value());
    } else if (key == "--state-hz") {
      args.state_hz = std::stod(value());
    } else if (key == "--watchdog-s") {
      args.watchdog_s = std::stod(value());
    } else if (key == "--velocity-tracking-gain") {
      args.velocity_tracking_gain = std::stod(value());
    } else if (key == "--min-x") {
      args.workspace.minimum[0] = std::stod(value());
    } else if (key == "--max-x") {
      args.workspace.maximum[0] = std::stod(value());
    } else if (key == "--min-y") {
      args.workspace.minimum[1] = std::stod(value());
    } else if (key == "--max-y") {
      args.workspace.maximum[1] = std::stod(value());
    } else if (key == "--min-z") {
      args.workspace.minimum[2] = std::stod(value());
    } else if (key == "--max-z") {
      args.workspace.maximum[2] = std::stod(value());
    } else if (key == "--guard-margin") {
      for (double& margin : args.workspace.guard_margin) margin = std::stod(value());
    } else if (key == "--workspace-braking-acceleration") {
      args.workspace.braking_acceleration = std::stod(value());
    } else if (key == "--workspace-reaction-s") {
      args.workspace.reaction_time_s = std::stod(value());
    } else if (key == "--workspace-braking-buffer") {
      args.workspace.braking_buffer = std::stod(value());
    } else if (key == "--velocity-caps") {
      for (double& limit : args.limits.velocity) limit = std::stod(value());
    } else if (key == "--acceleration-caps") {
      for (double& limit : args.limits.acceleration) limit = std::stod(value());
    } else if (key == "--jerk-caps") {
      for (double& limit : args.limits.jerk) limit = std::stod(value());
    } else {
      throw std::runtime_error("Unknown option: " + key);
    }
  }
  if (args.policy_hz <= 0 || args.state_hz <= 0 || args.watchdog_s <= 0 ||
      args.stop_settle_s <= 0 || args.velocity_tracking_gain <= 0) {
    throw std::runtime_error("Invalid timing configuration");
  }
  for (size_t joint = 0; joint < kJoints; ++joint) {
    if (args.limits.velocity[joint] <= 0 || args.limits.acceleration[joint] <= 0 ||
        args.limits.jerk[joint] <= 0) {
      throw std::runtime_error("Velocity, acceleration, and jerk caps must be positive");
    }
  }
  for (size_t axis = 0; axis < 3; ++axis) {
    if (args.workspace.guard_margin[axis] < 0 ||
        args.workspace.minimum[axis] + 2.0 * args.workspace.guard_margin[axis] >=
            args.workspace.maximum[axis]) {
      throw std::runtime_error("Invalid workspace bounds or guard margin");
    }
  }
  if (args.workspace.braking_acceleration <= 0 || args.workspace.reaction_time_s < 0 ||
      args.workspace.braking_buffer < 0) {
    throw std::runtime_error("Invalid workspace braking configuration");
  }
  return args;
}

struct GuardResult {
  bool safe = true;
  StopReason reason = StopReason::kNone;
  uint8_t rejected_step = 0;
  std::array<std::array<double, 3>, kMaxSteps> predicted_tcp{};
  uint8_t predicted_count = 0;
};

std::array<double, kJoints> reference_target_at_age(
    const std::array<std::array<double, kJoints>, kMaxSteps>& plan,
    uint8_t steps,
    double plan_age_s,
    const Args& args) {
  std::array<double, kJoints> target{};
  const double action_position = std::max(0.0, plan_age_s * args.policy_hz);
  const size_t step = static_cast<size_t>(std::floor(action_position));
  if (step >= steps) return target;

  const double alpha = clamp(action_position - static_cast<double>(step), 0.0, 1.0);
  const size_t next_step = std::min<size_t>(step + 1, steps - 1);
  for (size_t joint = 0; joint < kJoints; ++joint) {
    const double interpolated =
        (1.0 - alpha) * plan[step][joint] + alpha * plan[next_step][joint];
    target[joint] = clamp(
        interpolated, -args.limits.velocity[joint], args.limits.velocity[joint]);
  }
  return target;
}

GuardResult check_reference_predicted_plan(
    const franka::Model& model,
    const GuardState& state,
    const std::array<std::array<double, kJoints>, kMaxSteps>& plan,
    uint8_t start_step,
    uint8_t steps,
    double initial_plan_age_s,
    const Args& args) {
  GuardResult result{};
  std::array<double, kJoints> predicted_q = state.q;
  std::array<double, kJoints> commanded_velocity = state.dq_d;
  std::array<double, kJoints> commanded_acceleration = state.ddq_d;
  std::array<double, kJoints> upper_velocity = args.limits.velocity;
  std::array<double, kJoints> lower_velocity{};
  for (size_t joint = 0; joint < kJoints; ++joint) {
    lower_velocity[joint] = -upper_velocity[joint];
  }

  constexpr double kSimulationDt = franka::kDeltaT;
  constexpr double kMaximumStoppingTailS = 1.5;
  const double plan_end_s = static_cast<double>(steps) / args.policy_hz;
  const double simulation_limit_s =
      std::max(0.0, plan_end_s - initial_plan_age_s) + kMaximumStoppingTailS;
  size_t next_sample_step = start_step + 1;

  for (double elapsed_s = 0.0; elapsed_s <= simulation_limit_s;
       elapsed_s += kSimulationDt) {
    const double plan_age_s = initial_plan_age_s + elapsed_s;
    const auto target = reference_target_at_age(plan, steps, plan_age_s, args);
    const auto previous_velocity = commanded_velocity;
    commanded_velocity = franka::limitRate(
        upper_velocity, lower_velocity, args.limits.acceleration, args.limits.jerk,
        target, commanded_velocity, commanded_acceleration);
    for (size_t joint = 0; joint < kJoints; ++joint) {
      commanded_acceleration[joint] =
          (commanded_velocity[joint] - previous_velocity[joint]) / kSimulationDt;
      predicted_q[joint] +=
          0.5 * (previous_velocity[joint] + commanded_velocity[joint]) * kSimulationDt;
      if (predicted_q[joint] < args.limits.q_min[joint] ||
          predicted_q[joint] > args.limits.q_max[joint]) {
        result.safe = false;
        result.reason = StopReason::kPredictedGuard;
        result.rejected_step = static_cast<uint8_t>(std::min<size_t>(
            steps - 1, std::max<size_t>(start_step,
                static_cast<size_t>(std::floor(plan_age_s * args.policy_hz)))));
        return result;
      }
    }

    const auto pose = model.pose(
        franka::Frame::kEndEffector, predicted_q, state.F_T_EE, state.EE_T_K);
    const std::array<double, 3> xyz{{pose[12], pose[13], pose[14]}};
    while (next_sample_step <= steps &&
           plan_age_s >= static_cast<double>(next_sample_step) / args.policy_hz &&
           result.predicted_count < kMaxSteps) {
      result.predicted_tcp[result.predicted_count++] = xyz;
      ++next_sample_step;
    }
    for (size_t axis = 0; axis < 3; ++axis) {
      const double lower =
          args.workspace.minimum[axis] + args.workspace.guard_margin[axis];
      const double upper =
          args.workspace.maximum[axis] - args.workspace.guard_margin[axis];
      if (xyz[axis] < lower || xyz[axis] > upper) {
        result.safe = false;
        result.reason = StopReason::kPredictedGuard;
        result.rejected_step = static_cast<uint8_t>(std::min<size_t>(
            steps - 1, std::max<size_t>(start_step,
                static_cast<size_t>(std::floor(plan_age_s * args.policy_hz)))));
        return result;
      }
    }

    const bool plan_finished = plan_age_s >= plan_end_s;
    const bool stopped =
        std::all_of(commanded_velocity.begin(), commanded_velocity.end(),
                    [](double value) { return std::abs(value) < 1e-4; }) &&
        std::all_of(commanded_acceleration.begin(), commanded_acceleration.end(),
                    [](double value) { return std::abs(value) < 1e-3; });
    if (plan_finished && stopped) return result;
  }

  result.safe = false;
  result.reason = StopReason::kPredictedGuard;
  result.rejected_step = static_cast<uint8_t>(steps - 1);
  return result;
}

GuardResult check_predicted_plan(
    const franka::Model& model,
    const GuardState& state,
    const std::array<std::array<double, kJoints>, kMaxSteps>& plan,
    uint8_t start_step,
    uint8_t steps,
    double initial_plan_age_s,
    const Args& args) {
  if constexpr (kReferenceController) {
    return check_reference_predicted_plan(
        model, state, plan, start_step, steps, initial_plan_age_s, args);
  }
  GuardResult result{};
  std::array<double, kJoints> predicted_q = state.q;
  const double dt = 1.0 / args.policy_hz;
  const auto current_pose =
      model.pose(franka::Frame::kEndEffector, predicted_q, state.F_T_EE, state.EE_T_K);
  std::array<double, 3> previous_xyz{{current_pose[12], current_pose[13], current_pose[14]}};
  for (size_t step = start_step; step < steps; ++step) {
    for (size_t joint = 0; joint < kJoints; ++joint) {
      const double bounded_velocity = clamp(
          plan[step][joint], -args.limits.velocity[joint], args.limits.velocity[joint]);
      predicted_q[joint] += bounded_velocity * dt;
      if (predicted_q[joint] < args.limits.q_min[joint] || predicted_q[joint] > args.limits.q_max[joint]) {
        result.safe = false;
        result.reason = StopReason::kPredictedGuard;
        result.rejected_step = static_cast<uint8_t>(step);
        return result;
      }
    }
    const auto pose = model.pose(franka::Frame::kEndEffector, predicted_q, state.F_T_EE, state.EE_T_K);
    const std::array<double, 3> xyz{{pose[12], pose[13], pose[14]}};
    result.predicted_tcp[result.predicted_count++] = xyz;
    for (size_t axis = 0; axis < 3; ++axis) {
      const double axis_velocity = (xyz[axis] - previous_xyz[axis]) / dt;
      const double lower = args.workspace.minimum[axis] + args.workspace.guard_margin[axis] +
                           braking_distance(-axis_velocity, args.workspace);
      const double upper = args.workspace.maximum[axis] - args.workspace.guard_margin[axis] -
                           braking_distance(axis_velocity, args.workspace);
      if (xyz[axis] < lower || xyz[axis] > upper) {
        result.safe = false;
        result.reason = StopReason::kPredictedGuard;
        result.rejected_step = static_cast<uint8_t>(step);
        return result;
      }
    }
    previous_xyz = xyz;
  }
  return result;
}

void publish_robot_snapshot(RobotSnapshot* snapshot, const franka::RobotState& state) {
  snapshot->sequence.fetch_add(1, std::memory_order_acq_rel);
  for (size_t joint = 0; joint < kJoints; ++joint) {
    snapshot->q[joint].store(state.q[joint], std::memory_order_relaxed);
    snapshot->dq_d[joint].store(state.dq_d[joint], std::memory_order_relaxed);
    snapshot->ddq_d[joint].store(state.ddq_d[joint], std::memory_order_relaxed);
  }
  for (size_t index = 0; index < 16; ++index) {
    snapshot->F_T_EE[index].store(state.F_T_EE[index], std::memory_order_relaxed);
    snapshot->EE_T_K[index].store(state.EE_T_K[index], std::memory_order_relaxed);
  }
  snapshot->sequence.fetch_add(1, std::memory_order_release);
}

bool load_robot_snapshot(const RobotSnapshot& snapshot, GuardState* state) {
  for (int attempt = 0; attempt < 4; ++attempt) {
    const uint64_t before = snapshot.sequence.load(std::memory_order_acquire);
    if (before == 0 || (before & 1U) != 0) continue;
    for (size_t joint = 0; joint < kJoints; ++joint) {
      state->q[joint] = snapshot.q[joint].load(std::memory_order_relaxed);
      state->dq_d[joint] = snapshot.dq_d[joint].load(std::memory_order_relaxed);
      state->ddq_d[joint] = snapshot.ddq_d[joint].load(std::memory_order_relaxed);
    }
    for (size_t index = 0; index < 16; ++index) {
      state->F_T_EE[index] = snapshot.F_T_EE[index].load(std::memory_order_relaxed);
      state->EE_T_K[index] = snapshot.EE_T_K[index].load(std::memory_order_relaxed);
    }
    const uint64_t after = snapshot.sequence.load(std::memory_order_acquire);
    if (before == after) return true;
  }
  return false;
}

void validator_loop(const franka::Model& model,
                    const Mailbox* input,
                    const RobotSnapshot* robot_snapshot,
                    ValidatedMailbox* output,
                    const Args* args,
                    std::atomic<bool>* done) {
  uint64_t observed_generation = 0;
  while (!done->load(std::memory_order_acquire) &&
         !g_stop.load(std::memory_order_relaxed)) {
    const uint64_t source_generation = input->generation.load(std::memory_order_acquire);
    if (source_generation == observed_generation) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    const int source_buffer = input->active_buffer.load(std::memory_order_acquire);
    const uint8_t steps = input->steps[source_buffer].load(std::memory_order_relaxed);
    const uint64_t sequence = input->sequence[source_buffer].load(std::memory_order_relaxed);
    const uint64_t observation_ns =
        input->observation_monotonic_ns[source_buffer].load(std::memory_order_relaxed);
    std::array<std::array<double, kJoints>, kMaxSteps> plan{};
    for (size_t step = 0; step < steps; ++step) {
      for (size_t joint = 0; joint < kJoints; ++joint) {
        plan[step][joint] = input->velocity[source_buffer][step][joint].load(
            std::memory_order_relaxed);
      }
    }
    if (input->generation.load(std::memory_order_acquire) != source_generation) continue;

    GuardState state{};
    if (!load_robot_snapshot(*robot_snapshot, &state)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }

    const uint64_t now_ns = monotonic_ns();
    const double age_s = now_ns > observation_ns ? (now_ns - observation_ns) * 1e-9 : 0.0;
    const uint8_t offset = static_cast<uint8_t>(std::min<double>(
        kMaxSteps, std::floor(age_s * args->policy_hz)));
    GuardResult guard{};
    if (offset >= steps) {
      guard.safe = false;
      guard.reason = StopReason::kStalePlan;
      guard.rejected_step = offset;
    } else {
      guard = check_predicted_plan(model, state, plan, offset, steps, age_s, *args);
    }

    const int write_buffer = 1 - output->active_buffer.load(std::memory_order_relaxed);
    for (size_t step = 0; step < kMaxSteps; ++step) {
      for (size_t joint = 0; joint < kJoints; ++joint) {
        output->velocity[write_buffer][step][joint].store(
            plan[step][joint], std::memory_order_relaxed);
      }
      for (size_t axis = 0; axis < 3; ++axis) {
        output->predicted_tcp[write_buffer][step][axis].store(
            guard.predicted_tcp[step][axis], std::memory_order_relaxed);
      }
    }
    uint8_t validated_steps = steps;
    if (!guard.safe && guard.reason == StopReason::kPredictedGuard &&
        guard.rejected_step > offset) {
      if constexpr (kReferenceController) {
        // A prefix is accepted only when its own complete rate-limited stop is
        // safe. The rejected action index alone is not a stopping guarantee.
        for (uint8_t prefix_steps = guard.rejected_step;
             prefix_steps > offset; --prefix_steps) {
          GuardResult prefix_guard = check_predicted_plan(
              model, state, plan, offset, prefix_steps, age_s, *args);
          if (prefix_guard.safe) {
            validated_steps = prefix_steps;
            guard = prefix_guard;
            break;
          }
        }
      } else {
        validated_steps = guard.rejected_step;
        guard.safe = true;
        guard.reason = StopReason::kNone;
      }
    }
    output->steps[write_buffer].store(validated_steps, std::memory_order_relaxed);
    output->checked_offset[write_buffer].store(offset, std::memory_order_relaxed);
    output->predicted_count[write_buffer].store(guard.predicted_count, std::memory_order_relaxed);
    output->rejected_step[write_buffer].store(guard.rejected_step, std::memory_order_relaxed);
    output->sequence[write_buffer].store(sequence, std::memory_order_relaxed);
    output->observation_monotonic_ns[write_buffer].store(observation_ns, std::memory_order_relaxed);
    output->source_generation[write_buffer].store(source_generation, std::memory_order_relaxed);
    output->reason[write_buffer].store(static_cast<uint32_t>(guard.reason), std::memory_order_relaxed);
    output->safe[write_buffer].store(guard.safe, std::memory_order_relaxed);
    output->active_buffer.store(write_buffer, std::memory_order_release);
    output->generation.fetch_add(1, std::memory_order_release);
    observed_generation = source_generation;
  }
}

void store_state_output(StateOutputMailbox* output,
                        const franka::RobotState& state,
                        uint64_t active_sequence,
                        uint32_t step,
                        uint32_t steps,
                        uint32_t activation_offset,
                        StopReason stop_reason) {
  output->sequence.fetch_add(1, std::memory_order_acq_rel);
  output->monotonic_ns.store(monotonic_ns(), std::memory_order_relaxed);
  for (size_t joint = 0; joint < kJoints; ++joint) {
    output->q[joint].store(state.q[joint], std::memory_order_relaxed);
    output->dq[joint].store(state.dq[joint], std::memory_order_relaxed);
  }
  output->tcp_xyz[0].store(state.O_T_EE[12], std::memory_order_relaxed);
  output->tcp_xyz[1].store(state.O_T_EE[13], std::memory_order_relaxed);
  output->tcp_xyz[2].store(state.O_T_EE[14], std::memory_order_relaxed);
  output->active_sequence.store(active_sequence, std::memory_order_relaxed);
  output->plan_step.store(step, std::memory_order_relaxed);
  output->plan_steps.store(steps, std::memory_order_relaxed);
  output->activation_offset.store(activation_offset, std::memory_order_relaxed);
  output->stop_reason.store(static_cast<uint32_t>(stop_reason), std::memory_order_relaxed);
  output->sequence.fetch_add(1, std::memory_order_release);
}

bool load_state_packet(const StateOutputMailbox& output, StatePacket* packet) {
  for (int attempt = 0; attempt < 4; ++attempt) {
    const uint64_t before = output.sequence.load(std::memory_order_acquire);
    if (before == 0 || (before & 1U) != 0) continue;
    packet->magic = kStateMagic;
    packet->version = kVersion;
    packet->monotonic_ns = output.monotonic_ns.load(std::memory_order_relaxed);
    for (size_t joint = 0; joint < kJoints; ++joint) {
      packet->q[joint] = output.q[joint].load(std::memory_order_relaxed);
      packet->dq[joint] = output.dq[joint].load(std::memory_order_relaxed);
    }
    for (size_t axis = 0; axis < 3; ++axis) {
      packet->tcp_xyz[axis] = output.tcp_xyz[axis].load(std::memory_order_relaxed);
    }
    packet->active_sequence = output.active_sequence.load(std::memory_order_relaxed);
    packet->plan_step = output.plan_step.load(std::memory_order_relaxed);
    packet->plan_steps = output.plan_steps.load(std::memory_order_relaxed);
    packet->activation_offset = output.activation_offset.load(std::memory_order_relaxed);
    packet->stop_reason = output.stop_reason.load(std::memory_order_relaxed);
    const uint64_t after = output.sequence.load(std::memory_order_acquire);
    if (before == after) return true;
  }
  return false;
}

void state_publisher_loop(int socket_fd,
                          const sockaddr_in& target,
                          const StateOutputMailbox* output,
                          std::atomic<bool>* done) {
  uint64_t observed_sequence = 0;
  while (!done->load(std::memory_order_acquire) ||
         observed_sequence != output->sequence.load(std::memory_order_acquire)) {
    const uint64_t current_sequence = output->sequence.load(std::memory_order_acquire);
    if (current_sequence == observed_sequence || (current_sequence & 1U) != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    StatePacket packet{};
    if (!load_state_packet(*output, &packet)) continue;
    sendto(socket_fd, &packet, sizeof(packet), MSG_DONTWAIT,
           reinterpret_cast<const sockaddr*>(&target), sizeof(target));
    observed_sequence = current_sequence;
  }
}

TelemetryEvent base_event(const franka::RobotState& state) {
  TelemetryEvent event{};
  event.actual_tcp = {{state.O_T_EE[12], state.O_T_EE[13], state.O_T_EE[14]}};
  return event;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    Mailbox mailbox;
    EventQueue events;
    std::atomic<bool> logger_done{false};
    const int command_socket = make_bound_socket(args.command_port);
    const int state_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (state_socket < 0) throw std::runtime_error("Could not create UDP state socket");
    sockaddr_in state_target{};
    state_target.sin_family = AF_INET;
    state_target.sin_port = htons(args.state_port);
    state_target.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    std::thread receiver(receiver_loop, command_socket, &mailbox);
    std::thread logger(event_logger_loop, &events, &logger_done);
    ThreadCleanup cleanup{command_socket, state_socket, receiver, logger, logger_done};

    std::cout << "[STREAM] protocol=v" << kVersion
              << " controller=" << kControllerName
              << " command=127.0.0.1:" << args.command_port
              << " state=127.0.0.1:" << args.state_port
              << " policy_hz=" << args.policy_hz << " state_hz=" << args.state_hz
              << " dry_run=" << args.dry_run << std::endl;
    if constexpr (kReferenceController) {
      const size_t cpu_count = available_cpu_count();
      std::cout << "[STREAM] available_cpus=" << cpu_count << std::endl;
      if (!args.dry_run && cpu_count < 2) {
        throw std::runtime_error(
            "Reference controller requires at least two allowed CPUs for FCI and validation");
      }
      std::cout << "[STREAM] reference guard simulates the complete 1 kHz "
                   "rate-limited plan and stop tail; commission recorded C104 before policy use."
                << std::endl;
      std::cout << "[STREAM] reference limiter jerk_caps=[";
      for (size_t joint = 0; joint < kJoints; ++joint) {
        std::cout << args.limits.jerk[joint]
                  << (joint + 1 == kJoints ? "]" : ",");
      }
      std::cout << std::endl;
    } else {
      std::cout << "[STREAM] workspace braking acceleration="
                << args.workspace.braking_acceleration
                << "m/s^2 reaction=" << args.workspace.reaction_time_s
                << "s buffer=" << args.workspace.braking_buffer << "m" << std::endl;
    }

    if (args.dry_run) {
      uint64_t observed_generation = 0;
      uint64_t active_sequence = 0;
      uint8_t plan_steps = 0;
      uint8_t plan_step = 0;
      uint8_t activation_offset = 0;
      StopReason stop_reason = StopReason::kNone;
      bool reported_state_send = false;
      uint64_t state_send_count = 0;
      auto next_state_time = std::chrono::steady_clock::now();
      auto next_step_time = next_state_time;
      const auto plan_period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(1.0 / args.policy_hz));
      const auto state_period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(1.0 / args.state_hz));
      while (!g_stop.load(std::memory_order_relaxed)) {
        const auto now = std::chrono::steady_clock::now();
        const uint64_t generation = mailbox.generation.load(std::memory_order_acquire);
        if (generation != observed_generation) {
          const int buffer = mailbox.active_buffer.load(std::memory_order_acquire);
          const uint64_t observation_ns =
              mailbox.observation_monotonic_ns[buffer].load(std::memory_order_relaxed);
          const uint64_t now_ns = monotonic_ns();
          const double age_s = now_ns > observation_ns ? (now_ns - observation_ns) * 1e-9 : 0.0;
          plan_steps = mailbox.steps[buffer].load(std::memory_order_relaxed);
          plan_step = static_cast<uint8_t>(std::floor(age_s * args.policy_hz));
          activation_offset = plan_step;
          active_sequence = mailbox.sequence[buffer].load(std::memory_order_relaxed);
          observed_generation = generation;
          if (plan_step >= plan_steps) {
            stop_reason = StopReason::kStalePlan;
          } else {
            const auto observation_time = std::chrono::steady_clock::time_point(
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::nanoseconds(observation_ns)));
            next_step_time = observation_time + plan_period * (static_cast<int>(plan_step) + 1);
            std::cout << "[STREAM] dry-run activated seq=" << active_sequence
                      << " offset=" << int(plan_step) << " steps=" << int(plan_steps) << std::endl;
          }
        }
        while (stop_reason == StopReason::kNone && plan_step < plan_steps && now >= next_step_time) {
          ++plan_step;
          next_step_time += plan_period;
        }
        if (mailbox.stop_requested.load(std::memory_order_acquire)) stop_reason = StopReason::kCommand;
        if (now >= next_state_time) {
          StatePacket packet{};
          packet.magic = kStateMagic;
          packet.version = kVersion;
          packet.monotonic_ns = monotonic_ns();
          packet.active_sequence = active_sequence;
          packet.plan_step = plan_step;
          packet.plan_steps = plan_steps;
          packet.activation_offset = activation_offset;
          packet.stop_reason = static_cast<uint32_t>(stop_reason);
          const ssize_t sent = sendto(
              state_socket, &packet, sizeof(packet), MSG_DONTWAIT,
              reinterpret_cast<const sockaddr*>(&state_target), sizeof(state_target));
          if (!reported_state_send) {
            std::cout << "[STREAM] dry-run first state send bytes=" << sent << std::endl;
            reported_state_send = true;
          }
          ++state_send_count;
          if (state_send_count == 60) {
            std::cout << "[STREAM] dry-run state sends=60" << std::endl;
          }
          next_state_time += state_period;
        }
        if (stop_reason != StopReason::kNone) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      StatePacket final_packet{};
      final_packet.magic = kStateMagic;
      final_packet.version = kVersion;
      final_packet.monotonic_ns = monotonic_ns();
      final_packet.active_sequence = active_sequence;
      final_packet.plan_step = plan_step;
      final_packet.plan_steps = plan_steps;
      final_packet.activation_offset = activation_offset;
      final_packet.stop_reason = static_cast<uint32_t>(stop_reason);
      sendto(state_socket, &final_packet, sizeof(final_packet), MSG_DONTWAIT,
             reinterpret_cast<const sockaddr*>(&state_target), sizeof(state_target));
      std::cout << "[STREAM] dry-run stopped reason=" << stop_reason_name(stop_reason) << std::endl;
      return 0;
    }

    std::cout << "[STREAM] Persistent 1 kHz FCI control. Keep E-stop reachable. Type START: ";
    std::string confirmation;
    std::getline(std::cin, confirmation);
    if (confirmation != "START") throw std::runtime_error("Aborted before FCI control");

    franka::Robot robot(args.robot_ip);
    robot.setCollisionBehavior(
        {{20, 20, 18, 18, 16, 14, 12}}, {{20, 20, 18, 18, 16, 14, 12}},
        {{20, 20, 18, 18, 16, 14, 12}}, {{20, 20, 18, 18, 16, 14, 12}},
        {{20, 20, 20, 25, 25, 25}}, {{20, 20, 20, 25, 25, 25}},
        {{20, 20, 20, 25, 25, 25}}, {{20, 20, 20, 25, 25, 25}});
    const franka::Model model = robot.loadModel();
    auto active = robot.startJointVelocityControl(
        research_interface::robot::Move::ControllerMode::kJointImpedance);

    RobotSnapshot robot_snapshot;
    StateOutputMailbox state_output;
    ValidatedMailbox validated_mailbox;
    std::atomic<bool> validator_done{false};
    std::atomic<bool> state_publisher_done{false};
    std::thread validator(
        validator_loop, std::cref(model), &mailbox, &robot_snapshot,
        &validated_mailbox, &args, &validator_done);
    std::thread state_publisher(
        state_publisher_loop, state_socket, std::cref(state_target),
        &state_output, &state_publisher_done);
    ValidatorCleanup validator_cleanup{validator_done, validator};
    StatePublisherCleanup state_publisher_cleanup{state_publisher_done, state_publisher};

    std::array<double, kJoints> commanded{};
    std::array<double, kJoints> commanded_acceleration{};
    std::array<double, kJoints> target{};
    std::array<std::array<double, kJoints>, kMaxSteps> plan{};
    std::array<std::array<double, kJoints>, kMaxSteps> candidate{};
    uint8_t plan_steps = 0;
    uint8_t plan_step = 0;
    uint8_t activation_offset = 0;
    uint64_t active_sequence = 0;
    bool has_received_plan = false;
    uint64_t observed_generation = 0;
    StopReason stop_reason = StopReason::kNone;
    const auto plan_period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / args.policy_hz));
    const auto state_period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / args.state_hz));
    auto next_step_time = std::chrono::steady_clock::now();
    auto next_state_time = next_step_time;
    auto exhausted_since = std::chrono::steady_clock::time_point{};
    auto stopping_since = std::chrono::steady_clock::time_point{};
    std::array<double, 3> previous_tcp{};
    std::array<double, 3> filtered_tcp_velocity{};
    bool have_previous_tcp = false;
    bool command_state_initialized = false;

    while (true) {
      auto [state, period] = active->readOnce();
      const double dt = std::max(0.0005, std::min(0.0020, period.toSec()));
      const auto now = std::chrono::steady_clock::now();
      if (!command_state_initialized) {
        if constexpr (kReferenceController) {
          commanded = state.dq_d;
          commanded_acceleration = state.ddq_d;
        }
        command_state_initialized = true;
      }
      const double velocity_filter_alpha = clamp(dt / 0.010, 0.0, 1.0);
      for (size_t axis = 0; axis < 3; ++axis) {
        const double tcp = state.O_T_EE[12 + axis];
        if (have_previous_tcp) {
          const double measured_velocity = (tcp - previous_tcp[axis]) / dt;
          filtered_tcp_velocity[axis] +=
              velocity_filter_alpha * (measured_velocity - filtered_tcp_velocity[axis]);
        }
        previous_tcp[axis] = tcp;
      }
      have_previous_tcp = true;
      const uint64_t generation = validated_mailbox.generation.load(std::memory_order_acquire);

      if (stop_reason == StopReason::kNone && generation != observed_generation) {
        const int buffer = validated_mailbox.active_buffer.load(std::memory_order_acquire);
        const uint64_t source_generation =
            validated_mailbox.source_generation[buffer].load(std::memory_order_relaxed);
        if (source_generation != mailbox.generation.load(std::memory_order_acquire)) {
          observed_generation = generation;
        } else {
          const uint8_t candidate_steps =
              validated_mailbox.steps[buffer].load(std::memory_order_relaxed);
          const uint64_t sequence =
              validated_mailbox.sequence[buffer].load(std::memory_order_relaxed);
          const uint64_t observation_ns =
              validated_mailbox.observation_monotonic_ns[buffer].load(std::memory_order_relaxed);
          const uint64_t now_ns = monotonic_ns();
          const double age_s = now_ns > observation_ns ? (now_ns - observation_ns) * 1e-9 : 0.0;
          const uint8_t offset = static_cast<uint8_t>(std::min<double>(
              kMaxSteps, std::floor(age_s * args.policy_hz)));
          TelemetryEvent event = base_event(state);
          event.sequence = sequence;
          event.steps = candidate_steps;
          event.activation_offset = offset;
          event.observation_age_ms = age_s * 1000.0;
          for (size_t step = 0; step < candidate_steps; ++step) {
            for (size_t joint = 0; joint < kJoints; ++joint) {
              candidate[step][joint] =
                  validated_mailbox.velocity[buffer][step][joint].load(std::memory_order_relaxed);
              event.raw_peak = std::max(event.raw_peak, std::abs(candidate[step][joint]));
              event.raw_joint_peak[joint] =
                  std::max(event.raw_joint_peak[joint], std::abs(candidate[step][joint]));
              if (std::abs(candidate[step][joint]) > args.limits.velocity[joint]) {
                ++event.clipping_count[joint];
              }
            }
          }
          observed_generation = generation;
          has_received_plan = true;
          if (offset >= candidate_steps) {
            event.type = EventType::kPlanRejected;
            event.reason = StopReason::kStalePlan;
            event.rejected_step = offset;
            events.push(event);
            stop_reason = StopReason::kStalePlan;
          } else {
            event.predicted_count =
                validated_mailbox.predicted_count[buffer].load(std::memory_order_relaxed);
            event.rejected_step =
                validated_mailbox.rejected_step[buffer].load(std::memory_order_relaxed);
            for (size_t step = 0; step < event.predicted_count; ++step) {
              for (size_t axis = 0; axis < 3; ++axis) {
                event.predicted_tcp[step][axis] =
                    validated_mailbox.predicted_tcp[buffer][step][axis].load(
                        std::memory_order_relaxed);
              }
            }
            const bool guard_safe =
                validated_mailbox.safe[buffer].load(std::memory_order_relaxed);
            const StopReason guard_reason = static_cast<StopReason>(
                validated_mailbox.reason[buffer].load(std::memory_order_relaxed));
            if (!guard_safe) {
              event.type = EventType::kPlanRejected;
              event.reason = guard_reason;
              events.push(event);
              stop_reason = guard_reason;
            } else {
              event.type = EventType::kPlanAccepted;
              events.push(event);
              plan = candidate;
              plan_steps = candidate_steps;
              plan_step = offset;
              activation_offset = offset;
              active_sequence = sequence;
              const auto observation_time = std::chrono::steady_clock::time_point(
                  std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                      std::chrono::nanoseconds(observation_ns)));
              next_step_time = observation_time + plan_period * (static_cast<int>(offset) + 1);
              exhausted_since = std::chrono::steady_clock::time_point{};
            }
          }
        }
      }

      if (stop_reason == StopReason::kNone) {
        if (mailbox.stop_requested.load(std::memory_order_acquire)) {
          stop_reason = StopReason::kCommand;
        } else if (g_stop.load(std::memory_order_relaxed)) {
          stop_reason = StopReason::kSignal;
        }
      }
      if (stop_reason == StopReason::kNone) {
        for (size_t joint = 0; joint < kJoints; ++joint) {
          if (state.q[joint] < args.limits.q_min[joint] || state.q[joint] > args.limits.q_max[joint]) {
            stop_reason = StopReason::kJointGuard;
            break;
          }
        }
      }
      if (stop_reason == StopReason::kNone) {
        for (size_t axis = 0; axis < 3; ++axis) {
          const double xyz = state.O_T_EE[12 + axis];
          double lower = args.workspace.minimum[axis] + args.workspace.guard_margin[axis];
          double upper = args.workspace.maximum[axis] - args.workspace.guard_margin[axis];
          if constexpr (!kReferenceController) {
            const double commanded_tcp_velocity = state.O_dP_EE_c[axis];
            const double speed_toward_lower = std::max(
                {0.0, -filtered_tcp_velocity[axis], -commanded_tcp_velocity});
            const double speed_toward_upper = std::max(
                {0.0, filtered_tcp_velocity[axis], commanded_tcp_velocity});
            lower += braking_distance(speed_toward_lower, args.workspace);
            upper -= braking_distance(speed_toward_upper, args.workspace);
          }
          if (xyz <= lower || xyz >= upper) {
            stop_reason = StopReason::kWorkspaceGuard;
            break;
          }
        }
      }

      if (stop_reason == StopReason::kNone && plan_step < plan_steps) {
        while (now >= next_step_time && plan_step < plan_steps) {
          ++plan_step;
          next_step_time += plan_period;
        }
        if (plan_step >= plan_steps && exhausted_since == std::chrono::steady_clock::time_point{}) {
          exhausted_since = now;
        }
      }
      if (stop_reason == StopReason::kNone && has_received_plan && plan_step >= plan_steps) {
        if (exhausted_since == std::chrono::steady_clock::time_point{}) exhausted_since = now;
        if (now - exhausted_since > std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(args.watchdog_s))) {
          stop_reason = StopReason::kWatchdog;
        }
      }

      target.fill(0.0);
      if (stop_reason == StopReason::kNone && plan_step < plan_steps) {
        const auto step_start = next_step_time - plan_period;
        const double alpha = clamp(
            std::chrono::duration<double>(now - step_start).count() * args.policy_hz, 0.0, 1.0);
        const size_t next_index = std::min<size_t>(plan_step + 1, plan_steps - 1);
        for (size_t joint = 0; joint < kJoints; ++joint) {
          target[joint] =
              (1.0 - alpha) * plan[plan_step][joint] + alpha * plan[next_index][joint];
        }
      }

      if constexpr (kReferenceController) {
        std::array<double, kJoints> bounded_target{};
        std::array<double, kJoints> lower_velocity{};
        for (size_t joint = 0; joint < kJoints; ++joint) {
          bounded_target[joint] =
              clamp(target[joint], -args.limits.velocity[joint], args.limits.velocity[joint]);
          lower_velocity[joint] = -args.limits.velocity[joint];
        }
        const auto previous_commanded = commanded;
        commanded = franka::limitRate(
            args.limits.velocity, lower_velocity, args.limits.acceleration,
            args.limits.jerk, bounded_target, commanded, commanded_acceleration);
        for (size_t joint = 0; joint < kJoints; ++joint) {
          commanded_acceleration[joint] =
              (commanded[joint] - previous_commanded[joint]) / franka::kDeltaT;
        }
      } else {
        for (size_t joint = 0; joint < kJoints; ++joint) {
          const double bounded_target =
              clamp(target[joint], -args.limits.velocity[joint], args.limits.velocity[joint]);
          const double requested_acceleration = clamp(
              args.velocity_tracking_gain * (bounded_target - commanded[joint]),
              -args.limits.acceleration[joint], args.limits.acceleration[joint]);
          const double max_acceleration_delta = args.limits.jerk[joint] * dt;
          commanded_acceleration[joint] += clamp(
              requested_acceleration - commanded_acceleration[joint],
              -max_acceleration_delta, max_acceleration_delta);
          commanded[joint] += commanded_acceleration[joint] * dt;
          commanded[joint] =
              clamp(commanded[joint], -args.limits.velocity[joint], args.limits.velocity[joint]);
        }
      }

      if (now >= next_state_time) {
        publish_robot_snapshot(&robot_snapshot, state);
        store_state_output(
            &state_output, state, active_sequence, plan_step, plan_steps,
            activation_offset, stop_reason);
        next_state_time += state_period;
      }

      if (stop_reason != StopReason::kNone && stopping_since == std::chrono::steady_clock::time_point{}) {
        TelemetryEvent event = base_event(state);
        event.type = EventType::kStop;
        event.reason = stop_reason;
        events.push(event);
        stopping_since = now;
      }
      if (stop_reason != StopReason::kNone &&
          std::all_of(commanded.begin(), commanded.end(),
                      [](double value) { return std::abs(value) < 1e-3; }) &&
          now - stopping_since > std::chrono::duration_cast<std::chrono::steady_clock::duration>(
              std::chrono::duration<double>(args.stop_settle_s))) {
        publish_robot_snapshot(&robot_snapshot, state);
        store_state_output(
            &state_output, state, active_sequence, plan_step, plan_steps,
            activation_offset, stop_reason);
        active->writeOnce(franka::MotionFinished(franka::JointVelocities(commanded)));
        break;
      }
      active->writeOnce(franka::JointVelocities(commanded));
    }
    logger_done.store(true, std::memory_order_release);
    if (logger.joinable()) logger.join();
    std::cout << "[STREAM] stopped cleanly reason=" << stop_reason_name(stop_reason) << std::endl;
  } catch (const franka::Exception& exception) {
    std::cerr << "[STREAM] libfranka error: " << exception.what() << std::endl;
    return 1;
  } catch (const std::exception& exception) {
    std::cerr << "[STREAM] error: " << exception.what() << std::endl;
    return 1;
  }
  return 0;
}
