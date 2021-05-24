# -*- coding: utf-8 -*-
"""
Time modulated antenna array analysis


For arbitrary arrays of antennas, with arbitrary switching functions but same 
global modulation period, work out for an arbitrary sample rate, the 
pahse and magnitude of the spectrum in a far-field cross cut, and the mode 
purity, hopefully evenutally figuring out how to automate selection of 
harmonics





History:
Initial TDD 19 May 2016

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.constants as const
from vec3 import *

def heavyside(start,stop,signal):
    #signal must be 1D, but don't check, for speed.
    #must handle two cases - start before stop, and stop before start
    # start before stop ____|''''''|_______ 
    # stop before start '''|_______|'''''''

    if (np.size(start) > 1):    
        start = start[0]

    if (np.size(stop) > 1):    
        stop = stop[0]

    if (start == stop):
        return{signal * 0} #assume 0% duty cycle (more obvious if unintended)

    length = np.size(signal);
    
    if (length < 2):
        return{signal * 0} 
    
    mask = signal * 0;
    begin = int(np.round(length * start));
    end = int(np.round(length * stop));
        
    if (stop > start):
        mask[begin:end] = 1        
    else:
        mask[0:end] = 1
        mask[begin:] = 1
       
    return(signal * mask)
    
def sine_mask(relative_period, offset_rad, signal):
    wt = 2. * np.pi * np.arange(np.size(signal)) / relative_period 
    masked_signal = np.sin(wt + offset_rad) * signal
    return(masked_signal)   
    
def test_sine_mask():
    relative_period = 30
    offset_rad = np.pi /4.
    num_pts = 215
    test_signal = sin_signal(relative_period, offset_rad, num_pts)   
    relative_period = 111
    offset_rad = 0
    masked_signal = sine_mask(relative_period, offset_rad, test_signal)
    plt.figure()
    plt.plot(test_signal)
    plt.plot(masked_signal)
    plt.show()
    
def percentage(val):
    return(np.abs(val)*100.)    

def heavyside_test():
    '''
    Check for fencepost errors by issuing constrained requests where
    the duty cycle can be matched exactly, for the given number of samples
    Check for sensitivity of sampling to number of samples by issuing
    unconstrained requests
    '''
    requested_fill_factor = np.array([]);
    fill_factor_1 = np.array([]);
    fill_factor_2 = np.array([]);
    
    for num_pts in np.arange(10,300):
        for i in np.arange(1):
            
            pts = np.random.rand(2)
            start = np.floor( np.min(pts) * num_pts) / num_pts
            stop =  np.ceil( np.max(pts) * num_pts) / num_pts
            if start == stop:
                continue;
             
            this_fill_factor_1 =  np.sum(heavyside(start,stop,np.ones(num_pts)))/(num_pts)
            this_fill_factor_2 =  np.sum(heavyside(stop,start,np.ones(num_pts)))/(num_pts)
            fill_factor = stop - start
            
            requested_fill_factor = np.append(requested_fill_factor, fill_factor)
            fill_factor_1 = np.append(fill_factor_1, this_fill_factor_1)
            fill_factor_2 = np.append(fill_factor_2, this_fill_factor_2)
        
    error1 = percentage(fill_factor_1 - requested_fill_factor)    
    error2 = percentage(fill_factor_2 - (1 - requested_fill_factor))
    plt.figure()
    plt.plot(error1)
    plt.plot(error2)
    plt.ylabel('Error (%)')
    plt.xlabel('Sample number')
    plt.title('Error in requested duty cycle of the mask, for constrained requests')    
    plt.show()        
    
    requested_fill_factor = np.array([]);
    fill_factor_1 = np.array([]);
    fill_factor_2 = np.array([]);
    for num_pts in np.arange(10,300):
        for i in np.arange(1):
            
            pts = np.random.rand(2)
            start = np.min(pts)
            stop =  np.max(pts)
            if start == stop:
                continue;
             
            this_fill_factor_1 =  np.sum(heavyside(start,stop,np.ones(num_pts)))/(num_pts)
            this_fill_factor_2 =  np.sum(heavyside(stop,start,np.ones(num_pts)))/(num_pts)
            fill_factor = stop - start
            
            requested_fill_factor = np.append(requested_fill_factor, fill_factor)
            fill_factor_1 = np.append(fill_factor_1, this_fill_factor_1)
            fill_factor_2 = np.append(fill_factor_2, this_fill_factor_2)
        
    error1 = percentage(fill_factor_1 - requested_fill_factor)    
    error2 = percentage(fill_factor_2 - (1 - requested_fill_factor))
    plt.figure()
    plt.plot(error1)
    plt.plot(error2)
    plt.ylabel('Error (%)')
    plt.xlabel('Sample number')
    plt.title('Error in requested duty cycle of the mask, for unconstrained requests')    
    plt.show()   



def sin_signal(relative_period, offset_rad, num_pts):
    '''
    if we want num_pts of a sin wave of frequency f_sin, with a period
    of T = relative_period * (1 / f_sample) then we needn't explicitly 
    pass in the sample_frequency
    '''
    wt = 2. * np.pi * np.arange(num_pts) / relative_period 
    signal = np.sin(wt + offset_rad)
    return(signal)

def test_sin_signal():
    signal1 = sin_signal(2, np.pi/4.,103) #choose a strange number of pts
    signal2 = sin_signal(25.1,0,103) #choose non integer relative_period
    signal3 = sin_signal(49.56,0,103)
    plt.figure()
    plt.plot(signal1)
    plt.plot(signal2)
    plt.plot(signal3)
    plt.show()

def test_masked_sin_signal():
    signal1 = sin_signal(2, np.pi/4.,103)  #choose a strange number of pts
    signal2 = sin_signal(25.1,0,103)
    signal3 = sin_signal(49.56,0,103)
    
    for i in np.arange(3):
        pts = np.random.rand(2)
        start = np.min(pts)
        stop =  np.max(pts)
        
        masked_signal1 = heavyside(start,stop, signal1)
        masked_signal2 = heavyside(start,stop, signal2)
        masked_signal3 = heavyside(start,stop, signal3)
        
        plt.figure()
        plt.plot(masked_signal1)
        plt.plot(masked_signal2)
        plt.plot(masked_signal3)
        plt.show()
        
        masked_signal1 = heavyside(stop,start, signal1)
        masked_signal2 = heavyside(stop,start, signal2)
        masked_signal3 = heavyside(stop,start, signal3)
        
        plt.figure()
        plt.plot(masked_signal1)
        plt.plot(masked_signal2)
        plt.plot(masked_signal3)
        plt.show()


def point(x,y,z):
    return(np.array([x,y,z]))

def append_point(pt_array, pt):
    return(np.vstack((pt_array,pt)))

def xc(pt):
    if np.size(pt)==3:
        return(pt[0])
    else:
        return(pt[:,0])

def yc(pt):
    if np.size(pt)==3:
        return(pt[1])
    else:return(pt[:,1])

def zc(pt):
    if np.size(pt)==3:
        return(pt[2])
    else:
        return(pt[:,2])

def dist(pt1,pt2):
    return(np.linalg.norm(pt2-pt1))

def test_point():
    #plot a spiral
    pt_list = np.array([0,0,0])
    
    tpi = 2. * np.pi

    for i in np.arange(100):
        pt_list = append_point(pt_list, np.array([np.sin(i/tpi), np.cos(i/tpi), i/100.]))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xc(pt_list), yc(pt_list), zc(pt_list),'-*')    
    plt.show()
    
    plt.figure()
    distances = np.array([])
    distances_rev = np.array([])
    for this_pt in pt_list:
        origin = np.array([0,0,0])
        distances = np.append(distances, dist(origin, this_pt))
        distances_rev = np.append(distances_rev, dist(this_pt, origin)) 
   
    plt.plot(distances,'b--', label='normal')
    plt.plot(distances_rev,'g:', label='reverse order')
    plt.ylabel('distance')
    plt.xlabel('pt number')
    plt.legend(loc = 'lower right')
    plt.show()


def apply_offset(distance_metres, sample_frequency, signal):
    sample_wavelength = const.c / sample_frequency
    total_samples_to_offset = np.round(distance_metres / sample_wavelength)
    reduced_samples_to_offset = np.mod(total_samples_to_offset,np.size(signal))
    
    length = np.size(signal)
    a = int(length - reduced_samples_to_offset)
    
    if (reduced_samples_to_offset > 0):     
         offset_signal = np.hstack((signal[a:],signal[0:a]))
         return(offset_signal) 
    else:
         return(signal)   

def samples_per_switch_period(sampling_frequency, switch_frequency):
    return(np.round(sampling_frequency/switch_frequency))    
    
def relative_period(sin_frequency,sample_frequency):
    return(sample_frequency / sin_frequency)    
    
def test_apply_offset():
    '''
    #test shift
    foo = [0,1,2,3,4]
    a = 2; np.hstack((foo[a:],foo[0:a]))
    '''
    freq_list = np.linspace(1,3,3)*1e9
    distance_list = [2.6,2.75,2.9]
    sample_frequency = 50e9
    dt = 1/sample_frequency
    dx = dt * const.c 
    switch_frequency = 100e6
    signal_size = samples_per_switch_period(sample_frequency, 
                                            switch_frequency)
    dx_list = np.arange(signal_size) * dx
    
    for distance in distance_list:

        pts = [0.25,0.75]
        start = np.min(pts)
        stop =  np.max(pts)
        plt.figure()
        
        y_space = 1. #for plotting
        vert_shift = 0
        
        for frequency in freq_list:

            rel_period = relative_period(frequency,sample_frequency)
          
            signal = sin_signal(rel_period,np.pi/8.,signal_size)
    
            masked_signal = heavyside(start,stop, signal)
    
            vert_shift = vert_shift + y_space 
            offset_signal = apply_offset(distance, sample_frequency, masked_signal)        
          
            plt.plot(dx_list, offset_signal+ vert_shift) 
        plt.xlabel('distance/m')
        plt.ylabel('masked signal, offset')        
        plt.show();
        
class Antenna:
     def __init__(self, position, start, stop):
         self.pos = position
         self.start = start
         self.stop = stop

class AntennaSine:
     def __init__(self, position, rel_period, offset):
         self.pos = position
         self.rel_period = rel_period
         self.offset = offset
'''
def antenna(location,start,stop):
    return([np.array([location,start,stop])])


def append_antenna(antenna_list,this_antenna):
    return(np.concatenate((antenna_list,this_antenna),axis=1))
'''    
def do_all_unit_tests():
    print('heavyside test')
    heavyside_test()
    print('sin_signal test')    
    test_sin_signal()
    print('masked_sin_signal test') 
    test_masked_sin_signal()
    print('point test')
    test_point()
    print('apply_offset test')
    test_apply_offset();
    print('test mask sine')
    test_sine_mask()
if __name__ == '__main__':
    
    use_time_switching = True

    freq_list = np.linspace(1,3,3)*1e9
    distance_list = [2.6,2.75,2.9]
    sample_frequency = 1e12
    dt = 1/sample_frequency
    dx = dt * const.c 
    switch_frequency = 0.02e9

    signal_size = samples_per_switch_period(sample_frequency, 
                                            switch_frequency)
    frequency = freq_list[0] #loop later....
    rel_period = relative_period(frequency,sample_frequency)
    signal = sin_signal(rel_period,np.pi/8.,signal_size)

    num_ant = 8.
    zero_z = 0
    array_radius = 0.5 * const.c / np.min(freq_list)
    array = list()
    phi_list= (np.linspace(0,1,num_ant+1)*2. * np.pi)[:-1]
    start_list = np.arange(num_ant) / num_ant
    stop_list =  (np.arange(num_ant) + 1.) / num_ant
    for (phi, start, stop) in zip(phi_list, start_list, stop_list):    
        
        pos = vec3()
        pos.set_cylindrical(array_radius,phi,zero_z)
        array.append(Antenna(pos,start,stop))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for ant in array:
                
        plt.plot([ant.pos.x], [ant.pos.y], [ant.pos.z], 'r*')

   
    rx_z = 0.7
    rx_radius = 2
    do_line_rx = True
    line_rx = list() 
    if (do_line_rx == True):
        num_rx = 72
        xlist = np.linspace(-1,1,num_rx)
        for x in xlist:
            pos = vec3(x,0,rx_z)        
            line_rx.append(pos)
    else:
        num_rx = 36
        phi_list= (np.linspace(0,1,num_rx+1)*2. * np.pi)[:-1]
        for phi in phi_list:    
            pos = vec3()
            pos.set_cylindrical(rx_radius,phi,rx_z)
            line_rx.append(pos)    


    
    for rx in line_rx:
        plt.plot([rx.x], [rx.y], [rx.z], 'b+')
        
    # Comment or uncomment following both lines to test the fake bounding box:
    Xb = np.array([-1,1,-1,1,-1,1,-1,1])
    Yb = np.array([-1,-1,1,1,-1,-1,1,1])
    Zb = (np.array([-1,-1,-1,-1,1,1,1,1]) + 1)/2
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')        
        
    plt.show()
    
    rx_signals = np.zeros((int(num_rx),int(num_ant),int(signal_size)))
    rx_count = -1
    for rx in line_rx:
        rx_count = rx_count + 1
        ant_count = -1
        for ant in array:
            ant_count = ant_count + 1
            distance = abs(rx-ant.pos)
            if (use_time_switching == True):
                masked_signal = heavyside(ant.start,ant.stop, signal)
            else:
                masked_signal = heavyside(0,1, signal)    
            offset_signal = apply_offset(distance, sample_frequency, masked_signal) 
                 
            trim =  np.size(offset_signal) - np.size(rx_signals[rx_count, ant_count,:])     #hack for length change, fencepost error somewherE?      
            rx_signals[rx_count,ant_count,:] = (
                rx_signals[rx_count, ant_count,:] + 
                (offset_signal[trim:] / distance**2)) 
    '''
    plt.figure()
    for rx_num in np.arange(num_rx):            
            plt.plot(np.sum(rx_signals[rx_num,:,:] + rx_num, axis = 0))                    
    plt.show()
    '''
    
    '''
    rx_num = 6    
             
    plt.figure()

    for ant_num in np.arange(num_ant):
        plt.plot(rx_signals[rx_num,ant_num,:] + ant_num)

    
    fig.show()    
    plt.figure()
    '''
    rx_num = 6
    for ant_num in np.arange(num_ant):

        plt.plot(np.fft.fftshift(np.fft.fftfreq(2**15, 1/sample_frequency)),
                                20 * np.log10(np.fft.fftshift(np.abs(np.fft.fft(rx_signals[rx_num,ant_num,:],2**15)))))
        plt.xlim([0,2e9])                                
    
    fig.show()   
    
    plt.figure()
    phases0 = []
    phases1 = []
    phases2 = []
    phases3 = []
    
    amps0 = []
    amps1 = []
    amps2 = []
    amps3 = []
    
    for rx_num in np.arange(num_rx):
        fft_f = np.fft.fftshift(np.fft.fftfreq(2**19, 1/sample_frequency))
        fft_p = (np.fft.fftshift(np.angle(np.fft.fft(np.sum(rx_signals[rx_num,:,:], axis = 0),2**19))))
        fft_a = (20 * np.log10(np.fft.fftshift(np.abs(np.fft.fft(np.sum(rx_signals[rx_num,:,:], axis = 0),2**19)))))
        plt.plot(fft_f, fft_p)
        plt.xlim([0.94e9,1.06e9]) 
        jj = np.min(np.where(fft_f >= 1e9))                               
        phases0 = np.append(phases0, fft_p[jj])
        amps0 = np.append(amps0, fft_a[jj])
        
        jj = np.min(np.where(fft_f >= 0.98e9))                               
        phases1 = np.append(phases1, fft_p[jj])
        amps1 = np.append(amps1, fft_a[jj])

        jj = np.min(np.where(fft_f >= 0.96e9))                               
        phases2 = np.append(phases2, fft_p[jj])  
        amps2 = np.append(amps2, fft_a[jj])

        jj = np.min(np.where(fft_f >= 0.94e9))                               
        phases3 = np.append(phases3, fft_p[jj])
        amps3 = np.append(amps3, fft_a[jj])
        
    fig.show()   
    
   
    plt.figure()
    plt.plot(phases0)
    plt.plot(phases1 + 5)
    plt.plot(phases2 + 10)
    plt.plot(phases3 + 15)
    plt.show()
    
    plt.figure()
    plt.plot(amps0)
    plt.plot(amps1 )
    plt.plot(amps2 )
    plt.plot(amps3 )
    plt.show()
    '''
    plt.figure()
    plt.plot(np.sum(np.sum(rx_signals**2, axis = 2), axis = 1))    
 
    plt.show()     
    '''