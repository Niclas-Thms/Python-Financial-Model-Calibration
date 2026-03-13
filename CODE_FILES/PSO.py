import numpy as np

def particle_swarm(objective,population,max_iter=200, inertia=0.5,     
    attraction_personnal_best=1.5, attraction_global_best=2.0,    
    tol=1e-6, max_static_iter=10):

    swarm = np.array(population, dtype=float)
    n_particles, dim = swarm.shape
    velocities = np.zeros_like(swarm)

    # Personal bests
    pbest = swarm.copy()
    pbest_val = np.array([objective(x) for x in swarm])

    # Global best
    gbest_idx = np.argmin(pbest_val)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    static_iter, prev_best = 0, gbest_val
    for it in range(max_iter):
        for i in range(n_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            Up, Ug = attraction_personnal_best * r1, attraction_global_best * r2

            velocities[i] = (inertia * velocities[i] + Up * (pbest[i] - swarm[i]) + Ug * (gbest - swarm[i]))
            swarm[i] += velocities[i]
            value = objective(swarm[i])
            
            if value < pbest_val[i]: # Update personal best
                pbest[i] = swarm[i].copy()
                pbest_val[i] = value

                if value < gbest_val: # Update global best
                    gbest = swarm[i].copy()
                    gbest_val = value

        static_iter = static_iter + 1 if abs(prev_best - gbest_val) < tol else 0
        if static_iter >= max_static_iter:
            print(f"PSO stopped early at iteration {it}")
            break
        prev_best = gbest_val

    return gbest, gbest_val







