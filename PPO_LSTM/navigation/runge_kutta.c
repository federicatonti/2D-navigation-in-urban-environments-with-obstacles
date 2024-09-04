#include <math.h>
#include <stddef.h>

// Function to compute vorticity at a given position
double compute_vorticity(double* pos, double epsilon, 
                         double* flow_velocity_now, 
                         double* flow_velocity_x_plus, double* flow_velocity_x_minus, 
                         double* flow_velocity_y_plus, double* flow_velocity_y_minus) {
    // Compute partial derivatives using central differences
    double partial_v_x = (flow_velocity_y_plus[1] - flow_velocity_y_minus[1]) / (2 * epsilon);  // ∂v/∂x
    double partial_u_y = (flow_velocity_x_plus[0] - flow_velocity_x_minus[0]) / (2 * epsilon);  // ∂u/∂y

    // Vorticity is the difference between these partial derivatives
    double vorticity = partial_v_x - partial_u_y;
    
    return vorticity;
}

void dynamics(double* pos, double* vel, double theta, double omega, 
              double lin_acc, double ang_acc, double dt, 
              double* flow_velocity_now, double* flow_velocity_next, 
              double epsilon, 
              double* flow_velocity_x_plus, double* flow_velocity_x_minus, 
              double* flow_velocity_y_plus, double* flow_velocity_y_minus, 
              double* dx_dt, double* dy_dt, double* dvx_dt, double* dvy_dt, 
              double* dtheta_dt, double* domega_dt) {
    
    // Compute vorticity
    double vorticity = compute_vorticity(pos, epsilon, flow_velocity_now, 
                                         flow_velocity_x_plus, flow_velocity_x_minus, 
                                         flow_velocity_y_plus, flow_velocity_y_minus);
    
    // Compute flow field acceleration
    double flow_field_acceleration[2];
    flow_field_acceleration[0] = (flow_velocity_next[0] - flow_velocity_now[0]) / dt;
    flow_field_acceleration[1] = (flow_velocity_next[1] - flow_velocity_now[1]) / dt;
    
    // Effective velocity considering the flow field
    double effective_velocity[2];
    effective_velocity[0] = vel[0] + flow_velocity_now[0];
    effective_velocity[1] = vel[1] + flow_velocity_now[1];
    
    // Dynamics equations
    *dx_dt = effective_velocity[0];
    *dy_dt = effective_velocity[1];
    *dvx_dt = lin_acc * cos(theta) + flow_field_acceleration[0];
    *dvy_dt = lin_acc * sin(theta) + flow_field_acceleration[1];
    *dtheta_dt = omega;
    *domega_dt = ang_acc + vorticity;  // Include vorticity in the angular dynamics
}

void runge_kutta_4th(double* pos, double* vel, double theta, double omega, 
                     double lin_acc, double ang_acc, double dt, 
                     double* flow_velocity_now, double* flow_velocity_next, 
                     double epsilon, 
                     double* flow_velocity_x_plus, double* flow_velocity_x_minus, 
                     double* flow_velocity_y_plus, double* flow_velocity_y_minus, 
                     double* next_pos, double* next_vel, double* next_theta, double* next_omega) {
    
    double k1_pos[2], k1_vel[2], k1_theta, k1_omega;
    double k2_pos[2], k2_vel[2], k2_theta, k2_omega;
    double k3_pos[2], k3_vel[2], k3_theta, k3_omega;
    double k4_pos[2], k4_vel[2], k4_theta, k4_omega;
    
    // k1
    dynamics(pos, vel, theta, omega, lin_acc, ang_acc, dt, 
             flow_velocity_now, flow_velocity_next, epsilon, 
             flow_velocity_x_plus, flow_velocity_x_minus, 
             flow_velocity_y_plus, flow_velocity_y_minus, 
             &k1_pos[0], &k1_pos[1], &k1_vel[0], &k1_vel[1], 
             &k1_theta, &k1_omega);
    
    // k2
    double pos2[2], vel2[2], theta2, omega2;
    pos2[0] = pos[0] + 0.5 * k1_pos[0] * dt;
    pos2[1] = pos[1] + 0.5 * k1_pos[1] * dt;
    vel2[0] = vel[0] + 0.5 * k1_vel[0] * dt;
    vel2[1] = vel[1] + 0.5 * k1_vel[1] * dt;
    theta2 = theta + 0.5 * k1_theta * dt;
    omega2 = omega + 0.5 * k1_omega * dt;
    
    dynamics(pos2, vel2, theta2, omega2, lin_acc, ang_acc, dt, 
             flow_velocity_now, flow_velocity_next, epsilon, 
             flow_velocity_x_plus, flow_velocity_x_minus, 
             flow_velocity_y_plus, flow_velocity_y_minus, 
             &k2_pos[0], &k2_pos[1], &k2_vel[0], &k2_vel[1], 
             &k2_theta, &k2_omega);
    
    // k3
    double pos3[2], vel3[2], theta3, omega3;
    pos3[0] = pos[0] + 0.5 * k2_pos[0] * dt;
    pos3[1] = pos[1] + 0.5 * k2_pos[1] * dt;
    vel3[0] = vel[0] + 0.5 * k2_vel[0] * dt;
    vel3[1] = vel[1] + 0.5 * k2_vel[1] * dt;
    theta3 = theta + 0.5 * k2_theta * dt;
    omega3 = omega + 0.5 * k2_omega * dt;
    
    dynamics(pos3, vel3, theta3, omega3, lin_acc, ang_acc, dt, 
             flow_velocity_now, flow_velocity_next, epsilon, 
             flow_velocity_x_plus, flow_velocity_x_minus, 
             flow_velocity_y_plus, flow_velocity_y_minus, 
             &k3_pos[0], &k3_pos[1], &k3_vel[0], &k3_vel[1], 
             &k3_theta, &k3_omega);
    
    // k4
    double pos4[2], vel4[2], theta4, omega4;
    pos4[0] = pos[0] + k3_pos[0] * dt;
    pos4[1] = pos[1] + k3_pos[1] * dt;
    vel4[0] = vel[0] + k3_vel[0] * dt;
    vel4[1] = vel[1] + k3_vel[1] * dt;
    theta4 = theta + k3_theta * dt;
    omega4 = omega + k3_omega * dt;
    
    dynamics(pos4, vel4, theta4, omega4, lin_acc, ang_acc, dt, 
             flow_velocity_now, flow_velocity_next, epsilon, 
             flow_velocity_x_plus, flow_velocity_x_minus, 
             flow_velocity_y_plus, flow_velocity_y_minus, 
             &k4_pos[0], &k4_pos[1], &k4_vel[0], &k4_vel[1], 
             &k4_theta, &k4_omega);
    
    // Update the state using RK4 weighted average
    next_pos[0] = pos[0] + (k1_pos[0] + 2 * k2_pos[0] + 2 * k3_pos[0] + k4_pos[0]) / 6 * dt;
    next_pos[1] = pos[1] + (k1_pos[1] + 2 * k2_pos[1] + 2 * k3_pos[1] + k4_pos[1]) / 6 * dt;
    
    next_vel[0] = vel[0] + (k1_vel[0] + 2 * k2_vel[0] + 2 * k3_vel[0] + k4_vel[0]) / 6 * dt;
    next_vel[1] = vel[1] + (k1_vel[1] + 2 * k2_vel[1] + 2 * k3_vel[1] + k4_vel[1]) / 6 * dt;
    
    *next_theta = theta + (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6 * dt;
    *next_omega = omega + (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega) / 6 * dt;
}
