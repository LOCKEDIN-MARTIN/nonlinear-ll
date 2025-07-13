import numpy as np

import aero
from new_aerodata import *
from aero import *
from calc import *


if __name__ == '__main__':

    # wing parameters
    span = 1.48  # [m]
    root_c = 0.4  # [m]
    tip_c = 0.234  # [m]
    num_stations = 7  # make this an odd number to capture the root
    half_wing_chord = np.linspace(root_c, tip_c, int((num_stations+1)/2))
    chord = np.hstack((np.flip(half_wing_chord),half_wing_chord[1:]))
    stations = np.linspace(-span/2, span/2, num_stations)
    area = (root_c - tip_c)*span + tip_c*span

    # freestream parameters
    freestream = 10  # [m/s]
    Re_stations = aero.get_Re(1.225, freestream, chord, 0.00001837)
    alpha = 6  # [deg]

    # circulation calculation
    interpolator_input = np.column_stack((alpha*np.ones(np.shape(Re_stations)), Re_stations))
    c_l = interpolator(interpolator_input)
    g = gamma_dist(freestream, c_l, span/2, stations, chord, 1)  # initial guess assuming elliptical lift distribution
    dg_dx = -g*stations

    # induced angle of attack
    a_i = []
    for n in range(0, len(stations)):  # at each x_n
        a_i.append(1/(4*np.pi*freestream)*simpsons_with_singularity_fix(dg_dx, stations, stations[n]))

    # effective angle of attack
    a_eff = alpha*np.ones(np.shape(Re_stations)) - a_i
    
    # sectional lift coefficient
    interpolator_input = np.column_stack((a_eff*np.ones(np.shape(Re_stations)), Re_stations))
    c_l = interpolator(interpolator_input)

    # new circulation distribution
    g_old = g
    g_new = gamma_dist(freestream, c_l, span/2, stations, chord, 0)

    # circulation comparison
    tol = 1e-5  # accuracy requirement
    g_diff = np.linalg.norm(g_new-g_old)/np.linalg.norm(g_new)

    # circulation update
    D = 0.05  # damping coefficient
    g_input = g_old + D*(g_new - g_old)

    iter = 1

    while g_diff > tol or iter < 150:

        # circulation calculation
        g = g_input  # work with new input circulation distribution
        dg_dx = -g*stations

        # induced angle of attack
        a_i = []
        for n in range(0, len(stations)):  # at each x_n
            a_i.append(1/(4*np.pi*freestream)*simpsons_with_singularity_fix(dg_dx, stations, stations[n]))

        # effective angle of attack
        a_eff = alpha*np.ones(np.shape(Re_stations)) - a_i
        
        # sectional lift coefficient
        interpolator_input = np.column_stack((a_eff*np.ones(np.shape(Re_stations)), Re_stations))
        c_l = interpolator(interpolator_input)

        # new circulation distribution
        g_old = g
        g_new = gamma_dist(freestream, c_l, span/2, stations, chord, 0)

        # circulation comparison
        g_diff = np.linalg.norm(g_new-g_old)/np.linalg.norm(g_new)

        # circulation update
        g_input = g_old + D*(g_new - g_old)

        # iteration count
        iter += 1

    # lift and induced drag coefficients
    C_L = 2/(freestream*area)*np.trapz(g_new, stations, abs(stations[1]-stations[0]))
    C_Di = 2/(freestream*area)*np.trapz(g_new*a_i, stations, abs(stations[1]-stations[0]))

    #     c_l_sweep.append(aero.get_lift(freestream, area, gamma_new, stations))
    #     c_di_sweep.append(aero.get_induced_drag(aero.get_lift(freestream, area, gamma_new, stations), aspect_ratio, eff))

    # # ax1.plot(alpha_sweep, c_l_sweep)
    # # ax2.plot(alpha_sweep, c_di_sweep)

    # # ax1.set_title('Lift coefficient')
    # # ax2.set_title('Induced drag coefficient')
    # # ax1.set(ylabel='C_l')
    # # ax2.set(ylabel='C_di')
    # # fig.text(0.5, 0.04, 'Alpha [deg]', ha='center')
    # # ax1.grid()
    # # ax2.grid()
    # # plt.show()

    # fields = ['Alpha', 'C_di', 'C_l']

    # predict = []
    # for k in range(len(alpha_sweep)):
    #     predict.append([str(alpha_sweep[k]), str(c_di_sweep[k]), str(c_l_sweep[k])])

    # writename = 'output.csv'

    # with open(writename, 'w') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(fields)
    #     writer.writerows(predict)

    # csvfile.close()
