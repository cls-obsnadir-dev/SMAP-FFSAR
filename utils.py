#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def compute_cycle_pass(nb_cycle,nb_orbit,latitude):
    """
    Computes cycle/track information from cycle and orbit number (from Sentinel-3 L1a product name) and latitude (from the L1a netCDF file).
    """

    first_lat=latitude[0]
    last_lat=latitude[-1]
    new_cycle_number=nb_cycle

    if first_lat < last_lat:
        orbit_ascending=True
        new_pass_number=int(nb_orbit*2+1)
    else:
        orbit_ascending = False
        new_pass_number=int(nb_orbit*2)
    if new_pass_number == 771:
        new_pass_number=1
        new_cycle_number=nb_cycle+1

    pass_number=np.zeros(latitude.size,dtype=np.int)+new_pass_number
    cycle_number=np.zeros(latitude.size,dtype=np.int)+new_cycle_number

    if orbit_ascending:
        ind_min = int(np.nanargmin(latitude))
        ind_start_track_disc = range(ind_min)
        ind_max = int(np.nanargmax(latitude))
        if ind_max < latitude.shape[0] - 1:
            ind_end_track_disc = range(ind_max, latitude.shape[0], 1)
        else:
            ind_end_track_disc = []
    else:
        ind_max = int(np.nanargmax(latitude))
        ind_start_track_disc = range(ind_max)
        ind_min = int(np.nanargmin(latitude))
        if ind_min < latitude.shape[0] - 1:
            ind_end_track_disc = range(ind_min, latitude.shape[0], 1)
        else:
            ind_end_track_disc = []
    
    pass_number[ind_start_track_disc] -= np.int32(1)
    pass_number[ind_end_track_disc] += np.int32(1)

    if new_pass_number == 1:
        cycle_number[ind_start_track_disc] -= np.int32(1)
        pass_number[ind_start_track_disc] = 770

    if new_pass_number == 770:
        cycle_number[ind_end_track_disc] += np.int32(1)
        pass_number[ind_end_track_disc] = 1

    return new_cycle_number,cycle_number,new_pass_number,pass_number
