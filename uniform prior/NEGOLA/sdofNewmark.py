import jax.numpy as jnp
import jax
def sdofNewmark(m, c, k, delta_t, record, time, ui, uDoti, uDotDoti):
    g = 9.81
    beta = 1.0/6.0
    F1 = 1.0/beta*(m/delta_t+c/2.0)
    F2 = 1.0/2.0*(c*(2.0-1.0/(2.0*beta))*delta_t-m/beta)
    F3 = 1.0/beta*(m/delta_t**2+c/(2.0*delta_t))
    F4 = 1.0/(beta*delta_t**2)
    F5 = 1.0/(beta*delta_t)
    F6 = 1.0/(2.0*beta)
    F7 = 1.0/(2.0*beta*delta_t)
    F8 = 1.0/2.0*(2.0-1.0/(2.0*beta))*delta_t


    a_rel = jnp.zeros((len(record), 1))
    a_abs = jnp.zeros((len(record), 1))
    v = jnp.zeros((len(record), 1))
    d = jnp.zeros((len(record), 1))
    d = d.at[0].set(ui)
    v = v.at[0].set(uDoti)
    a_rel = a_rel.at[0].set(uDotDoti)
    a_abs = a_abs.at[0].set(uDotDoti+record[0])

    for i in range(len(record)-1):

        delta_force = -m * (record[i+1]-record[i])

        delta_displacement = (delta_force + F1*uDoti - F2*uDotDoti) / (F3 + jax.device_get(k))

        delta_velocity = F7*delta_displacement - F6*uDoti + F8*uDotDoti

        delta_acceleration = F4*delta_displacement - F5*uDoti - F6*uDotDoti

        ui += delta_displacement

        uDoti += delta_velocity

        uDotDoti += delta_acceleration

        time += delta_t

        d = d.at[i+1].set(ui)
        v = v.at[i+1].set(uDoti)
        a_abs = a_abs.at[i+1].set(uDotDoti + record[i+1])
        a_rel = a_rel.at[i+1].set(uDotDoti)

    print(1)
    return a_abs, v, d, a_rel