import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import copy
import warnings
import os

def prepared_error_dict(error_dict, config):
    if config.combine_gd and config.k != 1:
        warnings.warn(
            "Combine_gd has value 'True', but k != 1. Generator and discriminator losses will not be combined",
            RuntimeWarning
        )

    error_dict = copy.deepcopy(error_dict)
    if config.combine_sc:
        collapse_source_class(error_dict)
    if config.combine_rf or (config.combine_gd and config.k == 1): 
        dicts = [error_dict['D']['real'], error_dict['D']['fake']]
        error_dict['D'] = combine_dicts(dicts)
    if config.combine_gd and config.k == 1:
        dicts = [error_dict['D'], error_dict['G']]
        error_dict = combine_dicts(dicts)
    return error_dict


def plot_errors(to_plot, config, label=''):
    if type(to_plot) == list:
        xs = range(len(to_plot))
        if 'G' in label:
            xs = [x*config.k for x in xs]
        plt.plot(xs, to_plot, label=label)
    else:
        for key in to_plot.keys():
            plot_errors(to_plot[key], config, label+' '+key)


def save_error_plot(error_dict, config, nr=None):
    error_dict = prepared_error_dict(error_dict, config)
    plt.figure()

    fig_name = "Loss"
    if not nr is None:
        fig_name += ' of GAN ' + str(nr)

    plot_errors(error_dict, config)
    
    plt.xlabel('minibatch')
    plt.ylabel('loss')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(os.path.join(config.savefolder, 'loss'+str(nr)+'.png'))
    plt.close()
    
def collapse_source_class(error_dict):
    if 'source' in error_dict.keys():
        if 'classification' in error_dict.keys():
            dicts = [error_dict['source'], error_dict['classification']]
            error_dict = combine_dicts(dicts)
        else: 
            return
    else:
        for key in error_dict.keys():
            collapse_source_class(error_dict[key])

def combine_dicts(error_dicts):
    example = error_dicts[0]
    if type(example) == list:
        return [sum(x) for x in zip(*error_dicts)]
    #else type == dict
    return_dict = {}
    keys = example.keys()
    for key in keys:
        dicts = [d[key] for d in error_dicts]
        return_dict[key] = combine_dicts(dicts)
    return return_dict

def save_error_plots(error_dicts, config):
    if not config.combine_GANs:
        for nr, error_dict in enumerate(error_dicts):
            save_error_plot(error_dict, config, nr)
    else: 
        error_dict = combine_dicts(error_dicts)
        save_error_plot(error_dict, config)
