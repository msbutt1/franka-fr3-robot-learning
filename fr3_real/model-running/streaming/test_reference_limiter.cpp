#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

#include <franka/rate_limiting.h>

int main() {
  constexpr size_t kJoints = 7;
  const std::array<double, kJoints> velocity_limits{
      {0.41, 0.47, 0.025, 0.60, 0.14, 0.46, 0.39}};
  const std::array<double, kJoints> acceleration_limits{
      {1.75, 3.17, 0.34, 2.57, 0.85, 3.19, 1.77}};
  std::array<double, kJoints> lower_velocity{};
  std::array<double, kJoints> target{};
  std::array<double, kJoints> velocity{};
  std::array<double, kJoints> acceleration{};
  for (size_t joint = 0; joint < kJoints; ++joint) {
    lower_velocity[joint] = -velocity_limits[joint];
    target[joint] = (joint % 2 == 0 ? 0.8 : -0.8) * velocity_limits[joint];
  }

  for (size_t cycle = 0; cycle < 2000; ++cycle) {
    if (cycle == 250) target.fill(0.0);
    const auto previous_velocity = velocity;
    const auto previous_acceleration = acceleration;
    velocity = franka::limitRate(
        velocity_limits, lower_velocity, acceleration_limits,
        franka::kMaxJointJerk, target, velocity, acceleration);
    for (size_t joint = 0; joint < kJoints; ++joint) {
      acceleration[joint] =
          (velocity[joint] - previous_velocity[joint]) / franka::kDeltaT;
      const double jerk =
          (acceleration[joint] - previous_acceleration[joint]) / franka::kDeltaT;
      if (!std::isfinite(velocity[joint]) ||
          std::abs(velocity[joint]) > velocity_limits[joint] + 1e-9 ||
          std::abs(acceleration[joint]) > acceleration_limits[joint] + 1e-9 ||
          std::abs(jerk) > franka::kMaxJointJerk[joint] + 1e-6) {
        std::cerr << "Limiter violated a configured derivative at cycle " << cycle
                  << " joint " << joint << std::endl;
        return 1;
      }
    }
  }

  const double final_velocity = *std::max_element(
      velocity.begin(), velocity.end(),
      [](double left, double right) { return std::abs(left) < std::abs(right); });
  if (std::abs(final_velocity) >= 1e-4) {
    std::cerr << "Limiter failed to settle after a zero command: "
              << final_velocity << std::endl;
    return 1;
  }
  std::cout << "reference limiter derivative and settling test passed" << std::endl;
  return 0;
}
