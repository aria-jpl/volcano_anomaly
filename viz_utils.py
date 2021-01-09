import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_timeseries2d(timeseries, dates, suptitle, savepath):
    min_ts = np.nanmin(timeseries)
    max_ts = np.nanmax(timeseries)
    n_frames = timeseries.shape[0]
    fr_shape = timeseries.shape[1:]
    n_subplt_perfig = 25
    n_plt_per_axis = int(np.sqrt(n_subplt_perfig))
    
    if (n_frames % n_subplt_perfig == 0):
        n_figs = int(n_frames / n_subplt_perfig)
    else:
        n_figs = int(np.ceil(n_frames / n_subplt_perfig))
    
    fig_n = 0
    n_y, n_x = 0, 0
    fig, axs = plt.subplots(n_plt_per_axis, n_plt_per_axis, figsize=(50, 40))
    
    for i, fr in enumerate(timeseries):
        t = dates[i]
        
        title = f"{t[:4]}-{t[4:6]}-{t[6:]}"

        if (n_x < n_plt_per_axis):
            cmap = axs[n_y, n_x].imshow(fr, cmap='jet', vmin=min_ts, vmax=max_ts, interpolation=None, aspect='auto')
            axs[n_y, n_x].set_title(title, fontweight="bold", size=40)
            n_x += 1

        else:
            n_y += 1
            n_x = 0

            if (n_y < n_plt_per_axis):
                cmap = axs[n_y, n_x].imshow(fr, cmap='jet', vmin=min_ts, vmax=max_ts, interpolation=None, aspect='auto')
                axs[n_y, n_x].set_title(title, fontweight="bold", size=40)
                n_x += 1
            else:
                cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
                cbar_ax.tick_params(labelsize=50) 
                fig.colorbar(cmap, cax=cbar_ax)

                fig.suptitle(suptitle, fontsize=50)
                plt.savefig(f"{savepath}_{fig_n}.png")
                fig, axs = plt.subplots(n_plt_per_axis, n_plt_per_axis, figsize=(50, 40))
                n_y, n_x = 0, 0
                fig_n += 1

                cmap = axs[n_y, n_x].imshow(fr, cmap='jet', vmin=min_ts, vmax=max_ts, interpolation=None, aspect='auto')
                axs[n_y, n_x].set_title(title, fontweight="bold", size=40)
                n_x += 1
                
    if (n_y != 0 or n_x != 0):
        cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        cbar_ax.tick_params(labelsize=50)
        fig.colorbar(cmap, cax=cbar_ax)
        
def get_coord_position(size, n_points):
    if (size == n_points):
        return list(np.arange(size))
    else:
        space = size // (n_points+1)
        rest = size % (n_points+1)
        
        if (rest % 2 == 0):
            init = int(space + (rest/2))
        else:
            init = int(space + np.ceil(rest/2))
        
        coord = []
        for i in range(0, n_points):
            coord.append((init + (i*space)))
        
    return coord
    
def get_ticks_and_labels(val_range, n_ticks, start_date_idx, dates):  
    xticks_pos = np.linspace(val_range[0], val_range[1], n_ticks, dtype='int')
    xticks_labels = [dates[i+start_date_idx] for i in xticks_pos]
    return xticks_pos, xticks_labels

def extract_series(timeseries, x_pos, y_pos):
    sampled_series = []
    # Lists useful for plotting
    x_coord, y_coord = [], []
    coord = []
    for y in y_pos: 
        for x in x_pos:
            sampled_series.append(timeseries[:, y, x])
            y_coord.append(y)
            x_coord.append(x)
            coord.append((x,y))
    return sampled_series, x_coord, y_coord, coord

def get_anomaly_points(serie, diff):
    mean = np.nanmean(diff)
    std = np.nanstd(diff)
    x_points, y_points = [], []
    for i, s in enumerate(diff):
        if (s > abs(mean + mean)):
            x_points.append(i)
            y_points.append(serie[i])
    return x_points, y_points
            

def plot_timeseries1d(timeseries, dates, n_xaxis, n_yaxis, include_trend=False): 
    ysize, xsize = timeseries.shape[1], timeseries.shape[2]
    x_pos, y_pos = [], []
    
    # Extracting series
    assert (xsize >= n_xaxis)
    assert (ysize >= n_yaxis)
    x_pos = get_coord_position(xsize, n_xaxis)
    y_pos = get_coord_position(ysize, n_yaxis)
    sampled_series, x_coord, y_coord, coord = extract_series(timeseries, x_pos, y_pos)
            
    # Example frame for reference
    plt.figure()
    plt.title("Sampled Series")
    plt.plot(x_coord, y_coord, 'kX')
    plt.imshow(timeseries[-1, :, :], vmin=np.min(timeseries), vmax=np.max(timeseries), cmap='jet')
    plt.colorbar()
    plt.show()
    
    ## Plotting series side by side
    fig_n = 0
    n_y, n_x = 0, 0
    fig, axs = plt.subplots(n_yaxis, n_xaxis, figsize=(40, 40))
    nt = timeseries.shape[0]
    xticks_pos, xticks_labels = get_ticks_and_labels([0, nt-1], 5, 0, dates)
    
    assert (len(sampled_series) == (n_xaxis * n_yaxis))

    series_dict = dict()
    for i, serie in enumerate(sampled_series):
        if (n_x < n_xaxis):
            axs[n_y, n_x].plot(serie, 'b-', label='Real Values')
            mean_av = pd.DataFrame(serie).rolling(window=5).mean().values.reshape(-1)
            axs[n_y, n_x].plot(mean_av, 'r-', label='Moving Average')

            series_dict[(n_x, n_y)] = serie

            title = f"X={coord[i][0]}, Y={coord[i][1]}"
            axs[n_y, n_x].set_title(title, fontweight="bold", size=40)
            axs[n_y, n_x].set_xticks(xticks_pos)
            axs[n_y, n_x].set_xticklabels(xticks_labels)
            axs[n_y, n_x].set_ylabel('Displacement')
            axs[n_y, n_x].set_xlabel('Time Sample')
            handles, labels = axs[n_y, n_x].get_legend_handles_labels()
            axs[n_y, n_x].legend(handles, labels, loc='lower right')
            n_x += 1
        else:
            n_y += 1
            n_x = 0

            if (n_y < n_yaxis):
                axs[n_y, n_x].plot(serie, 'b-', label='Real Values')
                mean_av = pd.DataFrame(serie).rolling(window=5).mean().values.reshape(-1)
                axs[n_y, n_x].plot(mean_av, 'r-', label='Moving Average')
                
                series_dict[(n_x, n_y)] = serie

                title = f"X={coord[i][0]}, Y={coord[i][1]}"
                axs[n_y, n_x].set_title(title, fontweight="bold", size=40)
                axs[n_y, n_x].set_xticks(xticks_pos)
                axs[n_y, n_x].set_xticklabels(xticks_labels)
                axs[n_y, n_x].set_ylabel('Displacement')
                axs[n_y, n_x].set_xlabel('Time Sample')
                n_x += 1
            else:
                plt.show()
                break

    return sampled_series, series_dict


def plot_1dseries(data, dates):
    fig = plt.figure()
    
    for i, serie in enumerate(data['series']):
        assert (len(data['dates']) == serie.shape[0])
        plt.plot(serie, label=f"Normal Serie {i}")
    xticks_pos, xticks_labels = get_ticks_and_labels([0, len(serie)-1], 5, 0, dates)
    plt.xticks(xticks_pos, xticks_labels)
    plt.title('Normal Series')
    plt.xlabel('Time')
    plt.ylabel('Displacement')

    if data['label'] is not None:
        init_label = np.where(data['dates'] == data['label'][0])
        end_label = np.where(data['dates'] == data['label'][1])
        plt.axvline(x=init_label)
        plt.axvline(x=end_label)
        plt.title('Normal Series')
        plt.show()
