import taichi as ti
import os

ti.init(arch=ti.gpu)

max_num_particles = 256
curr_num_particles = 6
dt = 3e-3
num_particles = ti.field(ti.i32, shape=())

x = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
inv_m = ti.field(ti.f32, shape=max_num_particles)
x_proposed = ti.Vector(2, dt=ti.f32, shape=max_num_particles)
v = ti.Vector(2, dt=ti.f32, shape=max_num_particles)

# There are two constraint types in this simulation: 0 -- point constraint; 1 -- distance constraint;
# constraint_vector[c_idx][0] = c_type;
# constraint_vector[c_idx][1] = p1_idx;
# constraint_vector[c_idx][2] = p2_idx;
c_num = 6
constraint_vector = ti.Vector(3, dt=ti.int32, shape=max_num_particles)
gravity = ti.Vector([0, -9.8])
itr_num = 10

@ti.kernel
def init():
    # Init pos:
    for i in range(curr_num_particles):
        # p0 is the position constraint point;
        if i == 0:
            constraint_vector[i][0] = 0
            x[i] = [0.5, 0.8]
            inv_m[i] = 0.0
        else:
            x[i] = [0.5 - 0.1 * i, 0.8]
            inv_m[i] = 1
        print("x:", x[i])
    # Init constraints:
    for i in range(c_num):
        if i == 0:
            constraint_vector[i][0] = 0
        else:
            constraint_vector[i][0] = 1
            constraint_vector[i][1] = i - 1
            constraint_vector[i][2] = i


@ti.kernel
def handle_external_force():
    for i in range(curr_num_particles):
        # Handle gravity:
        v[i] = v[i] + dt * inv_m[i] * gravity
        # Damping velocity:
        v[i] = 0.99 * v[i]


@ti.kernel
def get_proposed_pos():
    for i in range(curr_num_particles):
        x_proposed[i] = x[i] + dt * v[i]


@ti.kernel
def project_constraints():
    for c_idx in range(c_num):
        # Position constraint:
        if constraint_vector[c_idx][0] == 0:
            p_idx = constraint_vector[c_idx][1]
            x_proposed[p_idx] = [0.5, 0.8]
        # Distance constraint:
        else:
            p1_idx = constraint_vector[c_idx][1]
            p2_idx = constraint_vector[c_idx][2]
            p1 = x_proposed[p1_idx]
            p2 = x_proposed[p2_idx]
            n = (p1 - p2).normalized()
            c = (p1 - p2).norm() - 0.1
            inv_m1 = inv_m[p1_idx]
            inv_m2 = inv_m[p2_idx]
            inv_m_sum = inv_m1 + inv_m2
            delta_p1 = - (inv_m1 * c / inv_m_sum) * n
            delta_p2 = (inv_m2 * c / inv_m_sum) * n
            x_proposed[p1_idx] += delta_p1
            x_proposed[p2_idx] += delta_p2


@ti.kernel
def update_vel_pos():
    for i in range(curr_num_particles):
        v[i] = (x_proposed[i] - x[i]) / dt
        x[i] = x_proposed[i]


# Main loop:
gui = ti.GUI('PBD', res=(512, 512), background_color=0xdddddd)
init()
num_particles[None] = 6
wait = input("PRESS ENTER TO CONTINUE.")
for frame in range(10000):
    # PBD steps:
    # Handle external force:
    handle_external_force()
    # Explicit Euler gets proposed pos:
    get_proposed_pos()
    # Generate collision constraints: No need :)
    # project constraints:
    for i in range(itr_num):
        project_constraints()
    # Update velocity and pos:
    update_vel_pos()

    # Render:
    X = x.to_numpy()
    gui.circles(X[:num_particles[None]], color=0xffaa77, radius=5)
    # Draw distance constraint:
    for i in range(c_num):
        if constraint_vector[i][0] == 1:
            idx1 = constraint_vector[i][1]
            idx2 = constraint_vector[i][2]
            gui.line(begin=X[idx1], end=X[idx2], radius=2, color=0x445566)
    gui.show()  # export and show in GUI
