# Copyright 2024 Crown in Right of Canada
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# To allow for more flexible development in loss functions, let's set the loss-relevant command-line
# parameters here.  We'll first set out the set of parameters as needed by argparse, specifiying:
# * parameter name
# * argparse variable name
# * datatype
# * default value
# * help string

from collections import namedtuple
LossParameter = namedtuple('LossParameter',
                           ('name','varname','dtype','default','help')
                           )

    
    # error_group.add_argument('--error-weights',type=str,dest='error_weight_file',default=None,
    #                     help='File containing non-default variable and level weights')
    # error_group.add_argument('--wind-speed',action='store_true',dest='wind_speed',default=False,
    #                     help='Add wind speed variable to loss function')
    # error_group.add_argument('--time-bias',action='store_true',dest='time_bias',default=False,
    #                     help='Add time-averaged term to loss function')
    # error_group.add_argument('--mean-bias',action='store_true',dest='mean_bias',default=False,
    #                     help='Add global mean bias term to loss function')
Parameters = (
    LossParameter('--error-weights','error_weight_file',str,None,'File containing non-default variable and level weights'),
    LossParameter('--wind-speed','wind_speed',bool,False,'Add wind speed variable to loss function'),
    LossParameter('--time-bias','time_bias',bool,False,'Add time-averaged term to loss function'),
    LossParameter('--mean-bias','mean_bias',bool,False,'Add global mean bias term to loss function'),
    LossParameter('--spectral-amse','spectral_amse',bool,False,'Compute loss in spectral space, with correlation/ampltidue adjustment'),
    LossParameter('--mae','mae',bool,False,'Compute loss with mean absolute error rather than MSE')
)

import argparse
def add_error_args(arggroup : argparse._ArgumentGroup | argparse.ArgumentParser):
    '''Given an argparse argument group, add the module-defined loss parameters.
    This function mutates the input argument.'''
    for p in Parameters:
        if (p.dtype is bool):
            # Boolean arguments get added with store_true or store_false
            if p.default is False:
                arggroup.add_argument(p.name,action='store_true',dest=p.varname,help=p.help)
            elif p.default is True:
                arggroup.add_argument(p.name,action='store_false',dest=p.varname,help=p.help)
            else:
                raise ValueError(f'Invalid default {p.default} for boolean parameter {p.name}')
        else:
            # Otherwise, add the argument generically.  Note that this suppots only scalar arguments.
            arggroup.add_argument(p.name,type=p.dtype,dest=p.varname,help=p.help)

config_dict = {} # Dictionary for loss-function configuration, to be populated by parse_arguments

def parse_arguments(parser : argparse.ArgumentParser):
    '''After the arguments have been read from the command line with parser.parse_args(),
    read the loss-relevant parameters and set the appropriate configuration dictionary entries'''

    args = vars(parser)
    for p in Parameters:
        config_dict[p.varname] = args[p.varname]

def normalize(ds,base,mean_by_level,stddev_by_level,diffs_stddev_by_level):
    #delta = ds - base
    import xarray as xr
    delta = xr.Dataset()
    for var in ds.data_vars:
        if (var in base.data_vars):
            delta[var] = (ds[var] - base[var])/diffs_stddev_by_level[var]
        else:
            delta[var] = (ds[var] - mean_by_level[var])/stddev_by_level[var]
    return delta

def derived_variables(ds,compute_wind_speed):
    # Creates a copy of ds and adds derived variables, controlled by global
    # variables.  Current incarnation is wind speed, may expand with time
    ds_out = ds.copy()
    if (compute_wind_speed):
        ds_out['wind_speed'] = (ds_out['u_component_of_wind']**2 + ds_out['v_component_of_wind']**2)**0.5
        ds_out['10m_wind_speed'] = (ds_out['10m_u_component_of_wind']**2 + ds_out['10m_v_component_of_wind']**2)**0.5
    return ds_out
                
# def losses_over_time(forecast,targets,analysis,per_variable_weights,norm_fn,compute_wind_speed=False):
#     from graphcast import xarray_jax
#     from graphcast import losses
#     forecast = derived_variables(forecast,compute_wind_speed)
#     targets = derived_variables(targets,compute_wind_speed)
#     analysis = derived_variables(analysis,compute_wind_speed)
#     norm_fc = norm_fn(forecast,analysis)
#     norm_tg = norm_fn(targets,analysis)
#     persist = 0*norm_fc.isel(time=0)
#     forecast_losses = [xarray_jax.unwrap_data(losses.weighted_mse_per_level(norm_fc.isel(time=lead),
#                                                                             norm_tg.isel(time=lead),
#                                                                             per_variable_weights)[0]) for lead in range(forecast.time.size)]
#     persist_losses = [xarray_jax.unwrap_data(losses.weighted_mse_per_level(persist,
#                                                                            norm_tg.isel(time=lead),
#                                                                            per_variable_weights)[0]) for lead in range(forecast.time.size)]
#     return(forecast_losses,persist_losses)

import typing
if typing.TYPE_CHECKING:
    # Imports required for type checking that should not be executed until
    # later in runtime.  Graphcast loads up the whole Jax infrastructure, and 
    # xarray loads pandas.
    import graphcast.graphcast
    import xarray

def get_model_coords(model_config):
    import xarray as xr
    import numpy as np
    model_latitude = xr.DataArray(np.linspace(-90,90,int(1+180/model_config['resolution']),dtype=np.float32),dims='latitude')
    model_latitude = model_latitude.assign_coords({'latitude' : model_latitude})
    model_longitude = xr.DataArray(np.linspace(0,360-model_config['resolution'],int(360/model_config['resolution']),dtype=np.float32),
                                dims='longitude')
    model_longitude = model_longitude.assign_coords({'longitude' : model_longitude})
    return (model_latitude, model_longitude)

def make_loss_new(model_config : 'graphcast.graphcast.ModelConfig', 
                  task_config : 'graphcast.graphcast.TaskConfig', 
                  diffs_stddev_by_level : typing.Optional['xarray.Dataset'],
                  mean_by_level : typing.Optional['xarray.Dataset'],
                  stddev_by_level : typing.Optional['xarray.Dataset'],
                  silent : bool = False):
    '''Using module-level configuration information supplied by parse_arguments(),
    build the required, differentiable, custom loss function loss(predictions,targets)
    that computes a MSE-like error function.
    
    This construction is much more flexible than the loss method built into graphcast,
    but it is slightly slower when taking gradients.  When there is no customization
    required, with no custom command-line arguments and None passed for the _by_level
    arguments, this function will return None to indicate that the standard loss function
    should be used instead.
    
    The optional parameter 'silent' suppresses printouts, intended for MPI invocation.
    '''

    # First, check if there is a need for a custom loss function
    if (diffs_stddev_by_level is None and \
        mean_by_level is None and \
        stddev_by_level is None and \
        all(config_dict[p.varname] == p.default for p in Parameters)):
        if (not silent): print('Using built-in (default) loss function')
        return None

    # Otherwise, build the supporting structure required for a custom loss function\
    import numpy as np
    import xarray as xr

    # First, latitude weights
    import graphcast.losses

    model_latitude, model_longitude = get_model_coords(model_config)
    latitude_weights = graphcast.losses.normalized_latitude_weights(model_latitude.rename(latitude='lat'))
    latitude_weights = latitude_weights / latitude_weights.mean()


    input_variables = list(task_config['input_variables'])
    target_variables = list(task_config['target_variables'])

    levels = np.array(task_config['pressure_levels'])

    # Next, level weights
    if config_dict['error_weight_file'] is not None:
        errfile_path = config_dict['error_weight_file']
        if not silent: print (f'Loading level and variable weights from {errfile_path}')

        with open(errfile_path,'rb') as errfile:
            import pickle
            (per_variable_weights, level_weights) = pickle.load(errfile)

            # Ensure all pressure levels are in the provided level weights
            assert(np.all(np.isin(levels,level_weights.level.data)))

            # Restrict the provided weights to only the selected levels,
            # for normalization
            level_weights = level_weights.sel(level=levels)
            level_weights = level_weights / level_weights.sum()
    else:
        if not silent: print('Using default level and variable weights')

        # Per variable weights are copied from Graphcast code
        per_variable_weights={
            "2m_temperature": 1.0,
            "10m_u_component_of_wind": 0.1,
            "10m_v_component_of_wind": 0.1,
            "mean_sea_level_pressure": 0.1,
            "total_precipitation_6hr": 0.1,
        }

        level_weights = xr.DataArray(levels,dims=('level'),coords={'level' : levels})
        level_weights = level_weights / level_weights.sum()

    if diffs_stddev_by_level is None:
        diffs_stddev_path = 'stats/diffs_stddev_by_level.nc'
        if not silent: print(f'Loading Δ6h standard deviation from {diffs_stddev_path}')
        diffs_stddev_by_level = xr.load_dataset(diffs_stddev_path).compute()

    if stddev_by_level is None:
        stddev_path = 'stats/stddev_by_level.nc'
        if not silent: print(f'Loading total standard deviation from {stddev_path}')
        stddev_by_level = xr.load_dataset(stddev_path).compute()

    # Combine diffs_stddev_by_level and stddev_by_level to form normalization factors,
    # selecitng from the first for output variables that are also input variables
    # and from the second for output varibles that are not input variables
    norms_by_level = xr.merge( [ diffs_stddev_by_level[v] if v in input_variables else stddev_by_level[v] \
                                    for v in target_variables ])

    # Check which if any extensions to the loss function computation are enabled
    # LossParameter('--wind-speed','wind_speed',bool,False,'Add wind speed variable to loss function'),
    # LossParameter('--time-bias','time_bias',bool,False,'Add time-averaged term to loss function'),
    # LossParameter('--mean-bias','mean_bias',bool,False,'Add global mean bias term to loss function'),
    # LossParameter('--spectral-amse','spectral_amse',bool,False,'Compute loss in spectral space, with correlation/ampltidue adjustment')

    compute_wind_speed = config_dict['wind_speed']
    time_bias = config_dict['time_bias']
    mean_bias = config_dict['mean_bias']
    spectral_amse = config_dict['spectral_amse']
    mae_error = config_dict['mae']

    if (spectral_amse and mae_error):
        raise ValueError('Cannot use both AMSE and MAE error calculations simultaneously')

    if (compute_wind_speed):
        if not silent: print('Computing loss function with wind speed added')
        norms_by_level = derived_variables(norms_by_level,compute_wind_speed)

    if (time_bias):
        if not silent: print('Computing loss function with additional time bias term')
    
    if (mean_bias):
        if not silent: print('Computing loss function with additional global mean loss term')

    if (spectral_amse): # Use spectral AMSE
        if not silent: print('Computing loss function in spectral space, with amplitude/correlation adjustment')
        # spectral AMSE is not compatible with time/mean bias loss
        if (time_bias or mean_bias):
            raise (ValueError('Spectral AMSE loss is not compatible with time-bias or mean-bias loss terms'))
        
        #import trainer.spectrum
        import graphcast.spectrum
        # Generate spectral coefficients
        if not silent: print('Generating spectral coefficients')
        leg_coef = graphcast.spectrum.generate_spectral_coefs(model_latitude)
        if not silent: print('... done')
        sht_forward = lambda f: graphcast.spectrum.sht_eval(leg_coef,f)
        
        def my_loss(prediction,targets):
            prediction = derived_variables(prediction,compute_wind_speed)
            targets = derived_variables(targets,compute_wind_speed)
            # The spectral loss is more sensitive to dimension ordering than the spatial loss,
            # so explicitly transpose the target array to match the prediction array
            targets = targets.transpose(*prediction.geopotential.dims)
            return spectral_adj_loss(prediction, targets,
                                 norms_by_level, sht_forward, 
                                 level_weights, per_variable_weights)
    elif mae_error: # Use MAE
        if not silent: print('Using mean absolute error for loss function')
        if (time_bias or mean_bias):
            raise (ValueError('MAE loss is not compatible with time-bias or mean-bias loss terms'))
        def my_loss(prediction,targets):
            prediction = derived_variables(prediction,compute_wind_speed)
            targets = derived_variables(targets,compute_wind_speed)
            return mae_loss(prediction,
                            targets,
                            per_variable_weights,
                            level_weights,
                            norms_by_level,
                            latitude_weights)

    else:
        def my_loss(prediction,targets):
            prediction = derived_variables(prediction,compute_wind_speed)
            targets = derived_variables(targets,compute_wind_speed)
            return spatial_loss(prediction,targets,
                                per_variable_weights,level_weights,
                                norms_by_level, latitude_weights, 
                                time_bias, mean_bias)
        
    return my_loss

def spectral_adj_loss(prediction : 'xarray.Dataset',analysis : 'xarray.Dataset',
                  norm_factors : 'xarray.Dataset',
                  sht_forward,
                  level_weights,per_variable_weights):
    import numpy as np
    import xarray as xr
    
    #from trainer.spectrum import sht_ds
    #from trainer.spectrum import power_spectral_density_ds
    #from trainer.spectrum import cross_spectral_density_ds
    from graphcast.spectrum import sht_ds
    from graphcast.spectrum import power_spectral_density_ds
    from graphcast.spectrum import cross_spectral_density_ds
    # Apply the normalization factors here, before taking the spectrum.
    # This makes the subsequent computations a bit better-conditioned. Otherwise
    # jax seems to have problems taking the gradient of loss with respect to 
    # stratospheric spectral humidity, which has a very very tiny scale
    pred_spec = sht_ds(prediction/norm_factors,sht_forward)
    targ_spec = sht_ds(analysis/norm_factors,sht_forward)
    pred_psd = power_spectral_density_ds(pred_spec)
    targ_psd = power_spectral_density_ds(targ_spec)
    # diff_psd = power_spectra(diff_spec)
    cross_psd = cross_spectral_density_ds(pred_spec,targ_spec)
    
    ## Update: if one of the prediction or analysis fields is exactly zero,
    ## then the derivative of the amplitude ratio is not well-defined, resulting in nans.
    ## By using max/min_psd explicitly, we can avoid the bothersome square root.
    max_psd = np.maximum(pred_psd,targ_psd)
    # min_psd = np.minimum(pred_psd,targ_psd)
    # variance_ratio = np.minimum(pred_psd,targ_psd)/(max_psd)
    # amplitude_ratio = variance_ratio**0.5
    
    # Rather than take √(A^2 P^2), add a little fudge factor to prevent division by zero
    geo_mean_psd = (1e-16 + pred_psd*targ_psd)**0.5
    corr_coef = np.minimum(1,cross_psd/geo_mean_psd)
    
    # raw_mse = max_psd * ( (1 - amplitude_ratio)**2 + 2*amplitude_ratio*(1-corr_coef))
    # raw_mse = variance_ratio*max_psd
    raw_mse = (pred_psd + targ_psd - 2*geo_mean_psd) + 2*max_psd*(1-corr_coef)
    
    # raw_mse_dc = xr.Dataset()
    
    # The 0-frequency is numerically special: the correlation is trivial, but numerically x^2 + y^2 - 2xy can
    # have catastrophic cancellation.  Better to write it directly as (x-y)^2 based on the 0-mode
    # for var in raw_mse.data_vars:
    dc_vars = {var: np.abs(pred_spec[var].isel(total_wavenumber=0,zonal_wavenumber=0) - \
                           targ_spec[var].isel(total_wavenumber=0,zonal_wavenumber=0))**2 for var in raw_mse.data_vars}
    dc_dset = xr.Dataset(dc_vars)
    # norm_mse = diff_psd / norm_factors**2
    norm_mse = raw_mse #/ norm_factors**2
    norm_dc = dc_dset #/ norm_factors**2
    
    mse_by_var = (norm_mse*level_weights).sum(dim='level').mean(dim=('time','batch'))
    dc_by_var = (norm_dc*level_weights).sum(dim='level').mean(dim=('time','batch'))

    loss_by_var = dc_by_var + mse_by_var.isel(total_wavenumber=slice(1,None)).sum(dim=('total_wavenumber'))
    # loss_by_var = mse_by_var.sum(dim=('total_wavenumber'))
    
    # Sum the dc component and the >0 wavenumber components together
    loss = sum(per_variable_weights.get(v,1.0) * loss_by_var[v] for v in mse_by_var.data_vars)
    return loss, loss_by_var

def spatial_loss(prediction,targets,per_variable_weights,level_weights,norms_by_level,latitude_weights,compute_time_bias,compute_mean_bias):
    # print('Compiling loss/gradient (inside spatial_loss)')
    # Remove any prediction variables that are not target variables.  This allows for slightly
    # nicer computation of mock errors, such as MSE(ic,target) for analysis persistence
    prediction = prediction[set(targets.data_vars)]
    diffs = targets - prediction
    # diffs = diffs/norm_by_level

    # Apply level weightings for both quadature and variable normalization at the same time
    # Note that norms_by_level needs to be squared here because it would otherwise apply
    # to diffs, not diffs**2
    adj_level_weights = level_weights / norms_by_level**2 

    # Construct the mean squared error from the inside out, based on likely variable ordering.
    # Jax's jit tends to exchange loops for the best order, but this helps speed non-jit calculations.
    mse = (((diffs**2).mean(dim='lon') * latitude_weights).mean(dim='lat')*adj_level_weights).\
            sum(dim='level').mean(dim=('time','batch'))
    # mse = (diffs**2 * latitude_weights * level_weights).sum(dim=('level')).mean(dim=('lat','lon','time','batch'))
    if (compute_time_bias and diffs.time.size > 1):
        # Use a small, currently hard-coded weight for the time-bias loss term
        # Take mean in time before squaring differences
        mse_time_bias = ((((diffs.mean(dim='time')**2).mean(dim='lon'))*latitude_weights).mean(dim='lat')*\
                        adj_level_weights).sum(dim='level').mean(dim='batch')
        mse = mse + 0.1*mse_time_bias
    if (compute_mean_bias):        # Take mean in lat/lon before squaring
        mse_mean_bias = (((diffs.mean(dim='lon') * latitude_weights).mean(dim='lat')**2)*adj_level_weights).sum(dim='level').mean(dim=('time','batch'))
        mse = mse + 0.1*mse_mean_bias
    total = sum((mse[i]*per_variable_weights.get(i,1.0) for i in mse.data_vars))
    return(total,mse)

def mae_loss(prediction,targets,per_variable_weights,level_weights,norms_by_level,latitude_weights):
    # print('Compiling loss/gradient (inside spatial_loss)')
    # Remove any prediction variables that are not target variables.  This allows for slightly
    # nicer computation of mock errors, such as MSE(ic,target) for analysis persistence
    import numpy as np
    from graphcast import xarray_jax
    prediction = prediction[set(targets.data_vars)]
    diffs = targets - prediction
    # diffs = diffs/norm_by_level

    # Apply level weightings for both quadature and variable normalization at the same time
    adj_level_weights = level_weights / norms_by_level

    # Construct the mean squared error from the inside out, based on likely variable ordering.
    # Jax's jit tends to exchange loops for the best order, but this helps speed non-jit calculations.
    mae = ((np.abs(diffs).mean(dim='lon') * latitude_weights).mean(dim='lat')*adj_level_weights).\
            sum(dim='level').mean(dim=('time','batch'))
    
    total = sum((mae[i]*per_variable_weights.get(i,1.0) for i in mae.data_vars))
    return(total,mae)

def unwrap_mean(f):
    '''A helper function, to take a loss function that returns a loss with some sort of structure
    (e.g. preserving time and batch dimensions), returning a wrapped function that takes the mean
    and removes the XArray structure in order to give a purely scalar loss.  This is performed 
    internally in build_loss_and_grad, for the all-in-one model-plus-loss/grad functions.'''
    from graphcast import xarray_jax
    from graphcast import xarray_tree
    def wrapped_f(*args):
        val = f(*args)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            val)
    return wrapped_f

def make_loss(norms_by_level,per_variable_weights,level_weights,latitude_weights,
              compute_wind_speed=False,compute_time_bias=False, compute_mean_bias=False):
    '''Define and return a custom loss function which uses user-specified standard deviations,
    per-variable weights, level weights, and latitude weights.  Using this inside the gradient
    calculation adds a small penalty because the losses and gradients are calculated in float32
    precision rather than bfloat16.
    
    The paramter norm_by_level gives the per-level, per-variable normalization weight to apply
    for the loss.  To match the GC calculation, this dataset should match diffs_stddev_by_level
    for all predicted variables that are also input variables and stddev_by_level for predicted
    variables that are not input variables (precipitation in GC-13).

    Parameters: 
        compute_wind_speed: derive wind speed and incorporate it into the MSE loss term
        compute_time_bias: add a separate term for Σ(mean_t(pred-target))^2, following G.4.3 
                           of the NeuralGCM preprint.  Note that this formulation requires
                           all times be available simultaneously, which is not necessarily the
                           case when running the validation code.
        compute_mean_bias: Add additional penalty term specifically for the global mean error, by
                           field and level
    '''
    def my_loss(prediction,targets):
        # print('Compiling loss/gradient (inside my_loss)')
        vars = list(targets.data_vars)
        prediction = derived_variables(prediction[vars],compute_wind_speed)
        targets = derived_variables(targets,compute_wind_speed)
        diffs = targets - prediction
        # diffs = diffs/norm_by_level

        # Apply level weightings for both quadature and variable normalization at the same time
        # Note that norms_by_level needs to be squared here because it would otherwise apply
        # to diffs, not diffs**2
        adj_level_weights = level_weights / norms_by_level**2 

        # Construct the mean squared error from the inside out, based on likely variable ordering.
        # Jax's jit tends to exchange loops for the best order, but this helps speed non-jit calculations.
        mse = (((diffs**2).mean(dim='lon') * latitude_weights).mean(dim='lat')*adj_level_weights).\
                sum(dim='level').mean(dim=('time','batch'))
        # mse = (diffs**2 * latitude_weights * level_weights).sum(dim=('level')).mean(dim=('lat','lon','time','batch'))
        if (compute_time_bias and diffs.time.size > 1):
            # Use a small, currently hard-coded weight for the time-bias loss term
            # Take mean in time before squaring differences
            mse_time_bias = ((((diffs.mean(dim='time')**2).mean(dim='lon'))*latitude_weights).mean(dim='lat')*\
                            adj_level_weights).sum(dim='level').mean(dim='batch')
            mse = mse + 0.1*mse_time_bias
        if (compute_mean_bias):        # Take mean in lat/lon before squaring
            mse_mean_bias = (((diffs.mean(dim='lon') * latitude_weights).mean(dim='lat')**2)*adj_level_weights).sum(dim='level').mean(dim=('time','batch'))
            mse = mse + 0.1*mse_mean_bias
        total = sum((mse[i]*per_variable_weights.get(i,1.0) for i in mse.data_vars))
        return(total,mse)

    return my_loss
